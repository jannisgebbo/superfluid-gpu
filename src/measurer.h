#ifndef MEASURER
#define MEASURER

#include "ModelA.h"
#include "nvector.h"
#include <array>
#include <complex>
#include <fftw3.h>
#include <memory>
#include <stdexcept>
#include <vector>
#ifdef SFGPU_CUFFT
#include <cufft.h>
#endif

////////////////////////////////////////////////////////////////////////
// Class to compute the fourier transform of the 1d array. The fourier transform
// is defined as 1/N \sum_x e^{ikx} W(x) = W(k) where W(x) is the array of size
// N and W(k) is the complex array of size N/2 + 1
class measurer_fft {

private:
  // Create the storage space for the plan
  std::vector<double> in_ptr;
  std::vector<std::complex<double>> out_ptr;
  fftw_plan plan;

public:
  measurer_fft(const size_t N) : in_ptr(N, 0.), out_ptr(N / 2 + 1, 0.) {
    // Create the plane for the fourier transform
    plan = fftw_plan_dft_r2c_1d(
        N, in_ptr.data(), reinterpret_cast<fftw_complex *>(out_ptr.data()),
        FFTW_MEASURE);
  }
  ~measurer_fft() { fftw_destroy_plan(plan); }

  void execute_row(double *in, std::complex<double> *out) {
    for (size_t i = 0; i < in_ptr.size(); i++) {
      in_ptr[i] = in[i];
    }
    fftw_execute(plan);
    for (size_t i = 0; i < out_ptr.size(); i++) {
      out[i] = out_ptr[i] / static_cast<double>(in_ptr.size());
    }
  }

  // Compute the fourier transform of an array in
  void execute(nvector<double, 2> &in, nvector<std::complex<double>, 2> &out) {
    for (size_t i = 0; i < in.N[0]; i++) {
      execute_row(&in(i, 0), &out(i, 0));
    }
  }
};

////////////////////////////////////////////////////////////////////////
// Class to compute a real-to-complex 3D fourier transform of a scalar field
// on a periodic box.  The transform is unnormalized,
//
//   W(k) = \sum_x e^{-i k.x} W(x),
//
// and callers divide by n0*n1*n2 (see Measurer::radialSpectrum).  The
// dimensions are (n0, n1, n2) in row-major (C) order, so that the last index
// n2 is contiguous in memory.  The real input has size n0*n1*n2 and the
// complex output (r2c half spectrum) has size n0*n1*(n2/2+1).
//
// execute_host transforms host memory with FFTW. With SFGPU_CUFFT,
// execute_device transforms device memory with cuFFT and leaves the result on
// the device. The plans and buffers are made on first use, so a GPU run never
// allocates the FFTW arrays and vice versa.
class measurer_fft_3d {

private:
  size_t n0, n1, n2;
  size_t out_size;
  // FFTW (host)
  std::vector<double> in_ptr;
  std::vector<std::complex<double>> out_ptr;
  fftw_plan fftw_plan_handle = nullptr;
#ifdef SFGPU_CUFFT
  // cuFFT (device)
  cufftHandle cufft_plan;
  bool have_cufft = false;
  cufftDoubleComplex *d_out = nullptr;
#endif

public:
  measurer_fft_3d(const size_t N0, const size_t N1, const size_t N2)
      : n0(N0), n1(N1), n2(N2), out_size(N0 * N1 * (N2 / 2 + 1)) {}

  ~measurer_fft_3d() {
    if (fftw_plan_handle) {
      fftw_destroy_plan(fftw_plan_handle);
    }
#ifdef SFGPU_CUFFT
    if (have_cufft) {
      cufftDestroy(cufft_plan);
      cudaFree(d_out);
    }
#endif
  }

  size_t size_in() const { return n0 * n1 * n2; }
  size_t size_out() const { return out_size; }

  // Transform the real volume `in` (host memory, size n0*n1*n2). Returns the
  // half spectrum in host memory, valid until the next call.
  const std::complex<double> *execute_host(const double *in) {
    if (!fftw_plan_handle) {
      in_ptr.assign(size_in(), 0.);
      out_ptr.assign(out_size, 0.);
      fftw_plan_handle = fftw_plan_dft_r2c_3d(
          static_cast<int>(n0), static_cast<int>(n1), static_cast<int>(n2),
          in_ptr.data(), reinterpret_cast<fftw_complex *>(out_ptr.data()),
          FFTW_MEASURE);
    }
    std::copy(in, in + size_in(), in_ptr.begin());
    fftw_execute(fftw_plan_handle);
    return out_ptr.data();
  }

#ifdef SFGPU_CUFFT
  // Transform the real volume `d_in` (device memory, size n0*n1*n2; not
  // modified). Returns the half spectrum in device memory, valid until the
  // next call.
  const cufftDoubleComplex *execute_device(const double *d_in) {
    if (!have_cufft) {
      if (cufftPlan3d(&cufft_plan, static_cast<int>(n0), static_cast<int>(n1),
                      static_cast<int>(n2), CUFFT_D2Z) != CUFFT_SUCCESS) {
        throw std::runtime_error("measurer_fft_3d: cufftPlan3d failed");
      }
      if (cudaMalloc(&d_out, out_size * sizeof(cufftDoubleComplex)) !=
          cudaSuccess) {
        cufftDestroy(cufft_plan);
        throw std::runtime_error("measurer_fft_3d: cudaMalloc failed");
      }
      have_cufft = true;
    }
    // An out-of-place D2Z transform leaves its input unchanged
    cufftExecD2Z(cufft_plan, const_cast<double *>(d_in), d_out);
    cudaDeviceSynchronize();
    return d_out;
  }
#endif
};

////////////////////////////////////////////////////////////////////////
// Computes slices of the fields and their fourier transforms
//
// The fields are labelled by U = phi, A, V, and phi2 and hence has dimensions
// NObs = Nphi +  NA +  NV  + 1 respectively.  The field phi2 is the square of
// the  field phi_a.
//
// Our fourier transform conventions are  (1/N_x) \sum_x e^{ikx} W(x) = W(k)
// where W(x)  is the average of the fields over the y and z directions, i.e.
// W(x) = 1/N_y * 1/N_z \sum_{y,z} U(x,y,z)
//
// The array wallX has dimensions NObs x N and contains the slices of the fields
// in the x direction. The array wallX_k has dimensions NObs x N/2 + 1 and
// contains the fourier transform of the slices of the fields in the x
// direction.
//
// The fields are rotated in the direction of the vev and the rotated fields are
// stored in the array wallX_rotated. The array wallXPhase has dimensions
// NObsPhase x N and contains the slices of the fields in the x direction of the
// rotated fields.  Finally the array wallXPhase_k has dimensions NObsPhase x
// N/2 + 1 and contains the fourier transform of the slices of the fields in the
// x direction of the rotated fields.
//
// The rotated fields are define as follows: The zero mode of the field phi
// defines a unit four vector n_a. The rotated fields are defined as follows:
//
// sigma = n_a phi_a,
// pi_b = phi_a - sigma n_b,
// A_a = rho_{ab} n_b,
// V_a = rhotilde_{ab} n_b,
// phi2.
//
// Here rho_{ab} is the field A_a and V_a are the
// axial and vector densities respectively and rhotilde_{ab} is the dual of the
// field rho_{ab}, i.e. multiplied by the epsilon tensor epsilon_{abcd} which
// swaps the vector and axial vector pieces.  Thus the dimension of the rotated
// fields is NObsPhase = 3*Nphi + 2 where Nphi is the number of scalar fields.
class Measurer {
public:
  // Arrays of size Nobs contain X=(phi[1..Nphi], q[1...Nq], phi2)
  static const PetscInt NObs =
      ModelAData::Nphi + ModelAData::NA + ModelAData::NV + 1;

  // Scalar data this is k=0 mode by itself
  static const PetscInt NScalars = NObs;
  std::vector<PetscScalar> OAverage;

  // Total energy of the ideal (Hamiltonian) sector,
  //   E = sum_x [ (A^2 + V^2) / (2 chi) + (grad phi)^2 / 2 - H phi_0 ],
  // with forward differences for grad phi. In superfluid mode H is the
  // effective field H * sigma/f used by the ideal step. Valid on rank 0 only.
  PetscScalar energy = 0.;

  // Spherically-averaged Fourier readout of the topological charge density
  // q(x) of the O(4) field (see computeTopChargeFourier in measurer.cxx).
  int NtopchargeBins = 0;
  double topcharge_dk = 0.;             // radial bin width = 2*pi/L
  std::vector<double> topcharge_Sk;     // S(k) = <|qtilde|^2> per radial shell
  std::vector<double> topcharge_kbins;  // |k| bin centers, = m*dk
  std::vector<double> topcharge_Nshell; // number of modes per shell
  std::complex<double> topcharge_zero{0., 0.}; // qtilde(k=0), DC component

  // Spherically-averaged power spectra of the fields from the full 3D grid
  // (see computeFieldSpectra in measurer.cxx), with the same |k| bins as the
  // topological charge. field_Sk is NFieldSpectra x NtopchargeBins (row
  // major): S_c(k) = <|U_c(k)|^2> per radial shell for U = (phi[4], A[3],
  // V[3]), with U_c(k) = (1/N^3) \sum_x e^{-ikx} U_c(x).
  static const PetscInt NFieldSpectra = ModelAData::Ndof;
  std::vector<double> field_Sk;
  std::vector<double> field_Nshell; // number of modes per shell

  // First dimension is NObs, last is spatial index x=0...N
  nvector<PetscScalar, 2> wallX;
  nvector<PetscScalar, 2> wallY;
  nvector<PetscScalar, 2> wallZ;

  // First dimension is NObs, last dimension is fourier index k=0..N/2+1
  nvector<std::complex<double>, 2> wallX_k;
  nvector<std::complex<double>, 2> wallY_k;
  nvector<std::complex<double>, 2> wallZ_k;

  // First dimension is NObsRotated, last dimension is fourier index.
  // Array of size NobsRotate contains XPhase = (sigma, pi[1..Nphi],
  // q[1..2*Nphi], phi2)
  static const PetscInt NObsRotated = 3 * ModelAData::Nphi + 2;
  nvector<std::complex<double>, 2> wallX_k_rotated;
  nvector<std::complex<double>, 2> wallY_k_rotated;
  nvector<std::complex<double>, 2> wallZ_k_rotated;

  // Array of size NobsPhase contains XPhase = (sigma, pi[1..Nphi],
  // q[1..2*Nphi], phi2)
  // The first dimension is NObsPhase, the second domenions is the spatial index
  static const PetscInt NObsPhase = 3 * ModelAData::Nphi + 2;
  nvector<PetscScalar, 2> wallXPhase;
  nvector<PetscScalar, 2> wallYPhase;
  nvector<PetscScalar, 2> wallZPhase;

  // First dimension is NObsPhase, last dimension is the fourier index 0..N/2+1
  nvector<std::complex<double>, 2> wallXPhase_k;
  nvector<std::complex<double>, 2> wallYPhase_k;
  nvector<std::complex<double>, 2> wallZPhase_k;

public:
  Measurer(ModelA *ptr) : model(ptr) {
    N = model->data.NX;
    if (model->data.NX != model->data.NY || model->data.NX != model->data.NZ) {
      PetscPrintf(
          PETSC_COMM_WORLD,
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
      throw(
          "Nx, Ny, and Nz must be equal for the correlation analysis to work");
    }

    // Only need the fft for derived observables
    int rank = -1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
       fftw = std::make_unique<measurer_fft>(N);
    }


    wallX.resize(NObs, N);
    wallY.resize(NObs, N);
    wallZ.resize(NObs, N);

    wallX_k.resize(NObs, N / 2 + 1);
    wallY_k.resize(NObs, N / 2 + 1);
    wallZ_k.resize(NObs, N / 2 + 1);

    wallX_k_rotated.resize(NObsRotated, N / 2 + 1);
    wallY_k_rotated.resize(NObsRotated, N / 2 + 1);
    wallZ_k_rotated.resize(NObsRotated, N / 2 + 1);

    wallXPhase.resize(NObsPhase, N);
    wallYPhase.resize(NObsPhase, N);
    wallZPhase.resize(NObsPhase, N);

    wallXPhase_k.resize(NObsPhase, N / 2 + 1);
    wallYPhase_k.resize(NObsPhase, N / 2 + 1);
    wallZPhase_k.resize(NObsPhase, N / 2 + 1);

    // Set up the radial binning for the spherically averaged topological
    // charge spectrum.  The bin width is the reciprocal-lattice spacing
    // dk = 2*pi/L (LX == LY == LZ here since NX == NY == NZ).  The largest
    // possible |k| integer magnitude is sqrt(3)*(N/2), so we need
    // floor(sqrt(3)*N/2 + 0.5)+1 bins with center |k|_m = m*dk.
    topcharge_dk = 2.0 * M_PI / model->data.LX;
    const double rmax = sqrt(3.0) * (0.5 * static_cast<double>(N));
    NtopchargeBins = static_cast<int>(floor(rmax + 0.5)) + 1;
    topcharge_Sk.assign(NtopchargeBins, 0.);
    topcharge_kbins.assign(NtopchargeBins, 0.);
    topcharge_Nshell.assign(NtopchargeBins, 0.);
    for (int m = 0; m < NtopchargeBins; m++) {
      topcharge_kbins[m] = m * topcharge_dk;
    }
    field_Sk.assign(NFieldSpectra * NtopchargeBins, 0.);
    field_Nshell.assign(NtopchargeBins, 0.);
    // The 3D FFT machinery is set up on first use (setupSpectra)
  }

  virtual ~Measurer() {
    VecScatterDestroy(&toZero);
    VecDestroy(&qseq);
    VecDestroy(&qnat);
    VecDestroy(&qglob);
    DMDestroy(&da1);
  }

  // Takes the vector, solution,  and computes the walls and their fourier
  // transforms and the rotated versions.  On output the arrays wallX, wallX_k,
  // wallX_rotated, as well as Y and Z are filled with the data. This can be
  // accessed to write the data to disk.
  void measure(Vec *solution) {
    int rank = -1;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    computeSliceAverages(solution);

    // Take the FFT and other steps based on the data collected
    if (rank == 0) {
      computeDerivedObs();
    }
  }

  // Computes the topological charge density q(x), its 3D FFT and the
  // spherically averaged power spectrum S(k) together with the zero mode.
  // computeTopChargeFourier is collective (it reduces q(x) to rank 0) so this
  // must be called on all ranks; the FFT/binning happens internally on rank 0.
  void measure_topcharge(Vec *solution) {
    computeTopChargeFourier(solution);
  }

  // Computes field_Sk, the radially binned 3D power spectra of all field
  // components. Collective, like measure_topcharge.
  void measure_field_spectra(Vec *solution) { computeFieldSpectra(solution); }

  ModelA *getModel() { return model; }
  PetscInt getN() { return N; }
  int getNTopchargeBins() { return NtopchargeBins; }
  int getNFieldSpectra() { return NFieldSpectra; }
  double getTopchargeDk() { return topcharge_dk; }

private:
  // Slice averages, phase slice averages and energy: on the device if the
  // solution is a Kokkos vector, otherwise with the host loops below
  void computeSliceAverages(Vec *solution);
  void computeSliceAverage(Vec *solution);
  void computeEnergy(G_node ***fld);
  void computeSliceAveragePhase(Vec *solution);
  void computeDerivedObs();
  void computeTopChargeFourier(Vec *solution);
  void computeFieldSpectra(Vec *solution);

  // 3D spectra of a scalar field q(x). q lives in qglob, a global vector of
  // the 1-dof DMDA da1 with the same layout as the model's domain.
  // radialSpectrum brings q to rank 0 in natural order (only a copy for one
  // rank), takes its FFT there (cuFFT if q is on the device) and bins
  // |q(k)|^2 into the |k| shells; Sk, Nshell and zero are set on rank 0.
  PetscErrorCode setupSpectra();
  PetscErrorCode radialSpectrum(double *Sk, double *Nshell,
                                std::complex<double> *zero);
  DM da1 = nullptr;
  Vec qglob = nullptr;
  Vec qnat = nullptr;           // natural ordering (more than one rank)
  Vec qseq = nullptr;           // all of q on rank 0 (more than one rank)
  VecScatter toZero = nullptr;

  ModelA *model;
  PetscInt N;

  // FFT engine using the fftw3 library
  std::unique_ptr<measurer_fft> fftw;
  // 3D FFT engine for the spectra (rank 0 only, made by setupSpectra)
  std::unique_ptr<measurer_fft_3d> fftw3d;
};

#endif
