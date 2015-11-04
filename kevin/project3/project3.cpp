#include "../utils.hpp"

double calc_elec_energy(arma::mat& P, arma::mat& H, arma::mat& F) {
  return arma::accu(P % (H + F));
}

void build_density(arma::mat& P, arma::mat& C, size_t NOcc) {
  P = C.cols(0, NOcc-1) * C.cols(0, NOcc-1).t();
}

void build_fock(arma::mat& F, arma::mat& P, arma::mat& H, arma::vec& ERI) {
  for (size_t mu = 0; mu < H.n_rows; mu++) {
   for (size_t nu = 0; nu < H.n_cols; nu++) {
      F(mu, nu) = H(mu, nu);
      for (size_t lm = 0; lm < P.n_rows; lm++) {
        for (size_t sg = 0; sg < P.n_cols; sg++) {
          F(mu, nu) += P(lm, sg) * (2*ERI(idx4(mu, nu, lm, sg)) - ERI(idx4(mu, lm, nu, sg)));
        }
      }
    }
  }
}

void trans_fock_to_prime(arma::mat& F_prime, arma::mat& F, arma::mat& X){
  F_prime = X.t() * F * X;
}
void trans_C_from_prime(arma::mat& C, arma::mat& C_prime, arma::mat& X){
  C = X * C_prime;
}
double rmsd_density(arma::mat& P_new, arma::mat& P_old) {
  return sqrt(arma::accu(arma::pow((P_new - P_old), 2)));
}

void mix_density(arma::mat& P_new, arma::mat& P_old, double alpha) {
  // alpha must be in the range [0, 1)
  P_new = ((1.0 - alpha) * P_new) + (alpha * P_old);
}

int main()
{

  size_t NElec = 10;
  size_t NOcc = NElec / 2;
  size_t NBasis = 7;
  size_t M = idx4(NBasis, NBasis, NBasis, NBasis);

  size_t i, j, k, l;
  double val;
  size_t mu, nu, lam, sig;

  FILE *enuc_file;
  enuc_file = fopen("h2o_sto3g_enuc.dat", "r");
  double Vnn;
  fscanf(enuc_file, "%lf", &Vnn);
  fclose(enuc_file);

  printf("Nuclear repulsion energy =  %12f\n", Vnn);

  arma::mat S(NBasis, NBasis);
  arma::mat T(NBasis, NBasis);
  arma::mat V(NBasis, NBasis);
  arma::mat H(NBasis, NBasis);
  arma::mat F(NBasis, NBasis, arma::fill::zeros);
  arma::mat F_prime(NBasis, NBasis, arma::fill::zeros);
  arma::mat D(NBasis, NBasis, arma::fill::zeros);
  arma::mat D_old(NBasis, NBasis, arma::fill::zeros);
  arma::mat D_new(NBasis, NBasis, arma::fill::zeros);
  arma::mat C(NBasis, NBasis);

  arma::vec eps_vec(NBasis);
  arma::mat C_prime(NBasis, NBasis);

  arma::vec Lam_S_vec(NBasis);
  arma::mat Lam_S_mat(NBasis, NBasis, arma::fill::zeros);
  arma::mat L_S(NBasis, NBasis);

  FILE *S_file, *T_file, *V_file;
  S_file = fopen("h2o_sto3g_s.dat", "r");
  T_file = fopen("h2o_sto3g_t.dat", "r");
  V_file = fopen("h2o_sto3g_v.dat", "r");

  while (fscanf(S_file, "%d %d %lf", &i, &j, &val) != EOF)
    S(i-1, j-1) = S(j-1, i-1) = val;
  while (fscanf(T_file, "%d %d %lf", &i, &j, &val) != EOF)
    T(i-1, j-1) = T(j-1, i-1) = val;
  while (fscanf(V_file, "%d %d %lf", &i, &j, &val) != EOF)
    V(i-1, j-1) = V(j-1, i-1) = val;

  fclose(S_file);
  fclose(T_file);
  fclose(V_file);

  arma::vec ERI = arma::vec(M, arma::fill::zeros);

  FILE *ERI_file;
  ERI_file = fopen("h2o_sto3g_eri.dat", "r");

  while (fscanf(ERI_file, "%d %d %d %d %lf", &i, &j, &k, &l, &val) != EOF) {
    mu = i-1; nu = j-1; lam = k-1; sig = l-1;
    ERI(idx4(mu,nu,lam,sig)) = val;
  }

  fclose(ERI_file);

  H = T + V;



  double thresh_E = 1.0e-15;
  double thresh_D = 1.0e-7;
  double d_E = 0.0;
  double d_D = 0.0;
  size_t iteration = 1;
  size_t max_iterations = 1024;
  double E_total, E_elec_old, E_elec_new, delta_E, rmsd_D;


  arma::eig_sym(Lam_S_vec, L_S, S);
  Lam_S_mat.diag() = Lam_S_vec;
  // Lam_S_mat = arma::diagmat(Lam_S_vec);
  arma::mat Lam_sqrt_inv = arma::sqrt(arma::inv(Lam_S_mat));
  arma::mat symm_orthog = L_S * Lam_sqrt_inv * L_S.t();

  // F = H;
  build_fock(F, D_new, H, ERI);
  trans_fock_to_prime(F_prime, F, symm_orthog);
  arma::eig_sym(eps_vec, C_prime, F_prime);
  C = symm_orthog * C_prime;
  build_density(D_new, C, NOcc);
  
  E_elec_new=calc_elec_energy(D_new, H, F);

  printf("Overlap Integrals:\n");
  print_arma_mat(S);
  printf("Kinetic-Energy Integrals:\n");
  print_arma_mat(T);
  printf("Nuclear Attraction Integrals\n");
  print_arma_mat(V);
  printf("Core Hamiltonian:\n");
  print_arma_mat(H);
  printf("S^-1/2 Matrix:\n");
  print_arma_mat(symm_orthog);
  printf("Initial F' Matrix:\n");
  print_arma_mat(F_prime);
  printf("Initial C Matrix:\n");
  print_arma_mat(C);
  printf("Initial Density Matrix:\n");
  print_arma_mat(D_new);
  
  printf("%6s %20s %20s %20s\n", "iter", "E_elec", "ddens", "dE");
  while (iteration < max_iterations) {
    E_elec_old = E_elec_new;
    D_old = D_new;
    build_fock(F, D_old, H, ERI);
    trans_fock_to_prime(F_prime, F, symm_orthog);
    arma::eig_sym(eps_vec, C_prime, F_prime);
    trans_C_from_prime(C, C_prime, symm_orthog);
    build_density(D_new, C, NOcc);
    E_elec_new=calc_elec_energy(D_new, H, F);

    //do stuff and test for convergence
    d_E = E_elec_new-E_elec_old;
    d_D = rmsd_density(D_new, D_old);
    printf("%6d %20.15f %20.15f %20.15f\n", iteration, E_elec_new, d_D, d_E);
    if ( d_E < thresh_E && d_D < thresh_D){
        break;
    }
    iteration++;
  }

  return 0;

}
