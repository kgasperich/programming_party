#include "../utils.hpp"

double calc_elec_energy(arma::mat& P, arma::mat& H, arma::mat& F) {
  return arma::accu(P % (H + F));
}

double calc_one_elec_prop(arma::mat& O, arma::mat& P) {
  return 2 * arma::accu(P % O);
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
void build_jk(arma::mat& J, arma::mat& K, arma::mat& P, arma::vec& ERI) {
  for (size_t mu = 0; mu < J.n_rows; mu++) {
   for (size_t nu = 0; nu < J.n_cols; nu++) {
      J(mu, nu) = 0;
      K(mu, nu) = 0;
      for (size_t lm = 0; lm < P.n_rows; lm++) {
        for (size_t sg = 0; sg < P.n_cols; sg++) {
          J(mu, nu) += P(lm, sg) * ERI(idx4(mu, nu, lm, sg));
          K(mu, nu) += P(lm, sg) * ERI(idx4(mu, lm, nu, sg));
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

void trans_ERI2(arma::vec& ERI_in, arma::vec& ERI_out, arma::mat& C) {
  int nbas = C.n_rows;
  for (int p = 0; p < nbas; p++){
    for (int q = 0; q <= p; q++){
      for (int r = 0; r <= p; r++){
        for (int s = 0; s <=r && (!(p==r) || s <=q); s++){
          for (int i = 0; i < nbas; i++){
            for (int j = 0; j < nbas; j++){
              for (int k = 0; k < nbas; k++){
                for (int l = 0; l < nbas; l++){
                  ERI_out(idx4(p,q,r,s)) += C(i,p) * C(j,q) * C(k,r) * C(l,s) * ERI_in(idx4(i,j,k,l));
                }
              }
            }
          }
        }
      }
    }
  }
}
void trans_ERI3(arma::vec& ERI_in, arma::vec& ERI_out, arma::mat& C) {
  int nbas = C.n_rows;
  arma::vec temp1(nbas);
  arma::mat temp2(nbas,nbas);
  arma::cube temp3(nbas,nbas,nbas);

  //for (int p = 0; p < nbas+1; p++){
  for (int p = 0; p < nbas; p++){
    temp3.zeros();
    for (int i = 0; i < nbas; i++){
      for (int j = 0; j < nbas; j++){
        for (int k = 0; k < nbas; k++){
          for (int l = 0; l < nbas; l++){
            temp3(j,k,l) += C(i,p) * ERI_in(idx4(i,j,k,l));
            // still need to exploit kl symmetry
          }
        }
      }
    }
    for (int q = 0; q <= p; q++){
      temp2.zeros();
      for (int j = 0; j < nbas; j++){
        for (int k = 0; k < nbas; k++){
          for (int l = 0; l < nbas; l++){
            temp2(k,l) += C(j,q) * temp3(j,k,l);
          }
        }
      }
      for (int r = 0; r <= p; r++){
        temp1.zeros();
        for (int k = 0; k < nbas; k++){
          for (int l = 0; l < nbas; l++){
            temp1(l) += C(k,r) * temp2(k,l);
          }
        }
        for (int s = 0; s <=r && (!(p==r) || s <=q); s++){
          ERI_out(idx4(p,q,r,s)) = dot(C.col(s), temp1);
  //        printf("%5d %4d %4d %4d %4d %4d %4d\n",idx4(p,q,r,s),idx2(p,q),idx2(r,s),p,q,r,s);
        }
      }
    }
  }
//  printf("%d\n",nbas);
}
void trans_ERI(arma::vec& ERI_in, arma::vec& ERI_out, arma::mat& C) {
  int nbas = C.n_rows;
  arma::vec temp1(nbas);
  arma::mat temp2(nbas,nbas);
  arma::cube temp3(nbas,nbas,nbas);

  //for (int p = 0; p < nbas+1; p++){
  for (int p = 0; p < nbas; p++){
    temp3.zeros();
    for (int i = 0; i < nbas; i++){
      for (int j = 0; j < nbas; j++){
        for (int k = 0; k < nbas; k++){
          for (int l = 0; l <= k; l++){
            temp3(j,k,l) += C(i,p) * ERI_in(idx4(i,j,k,l));
            // still need to exploit kl symmetry
          }
        }
      }
    }
    for (int q = 0; q <= p; q++){
      temp2.zeros();
      for (int j = 0; j < nbas; j++){
        for (int k = 0; k < nbas; k++){
          for (int l = 0; l <= k; l++){
            temp2(k,l) += C(j,q) * temp3(j,k,l);
          }
        }
      }
      printf("before trans\n");
      print_arma_mat(temp2);
      temp2=symmatl(temp2);
      printf("after trans\n");
      print_arma_mat(temp2);
      for (int r = 0; r <= p; r++){
        temp1.zeros();
        for (int k = 0; k < nbas; k++){
          for (int l = 0; l < nbas; l++){
            temp1(l) += C(k,r) * temp2(k,l);
          }
        }
        for (int s = 0; s <=r && (!(p==r) || s <=q); s++){
          ERI_out(idx4(p,q,r,s)) = dot(C.col(s), temp1);
  //        printf("%5d %4d %4d %4d %4d %4d %4d\n",idx4(p,q,r,s),idx2(p,q),idx2(r,s),p,q,r,s);
        }
      }
    }
  }
//  printf("%d\n",nbas);
}

double calc_mp2_energy(arma::vec& ERI_mo, arma::vec& e, size_t nocc, size_t nbas){
  double emp2 = 0.0;
  int iajb;
  for (int i = 0; i < nocc; i++){
    for (int j = 0; j < nocc; j++){
      for (int a = nocc; a < nbas; a++){
        for (int b = nocc; b < nbas; b++){
          iajb = idx4(i,a,j,b);
          emp2 += (ERI_mo(iajb) * (2 * ERI_mo(iajb) - ERI_mo(idx4(i,b,j,a))))/(e(i) + e(j) - e(a) - e(b));
        }
      }
    }
  }
  return emp2;
}

int main()
{
  size_t NElec = 10;
  size_t NOcc = NElec / 2;
  size_t NBasis = 7;
  size_t M2 = idx4(NBasis-1, NBasis-1, NBasis-1, NBasis-1);
  size_t M = idx4(NBasis, NBasis, NBasis, NBasis);
  size_t Nnuc ;
//  int Nnuc;
//  long unsigned Nnuc;
//  unsigned Nnuc;
  size_t i, j, k, l;
  double val, qi, xi, yi, zi;
  size_t mu, nu, lam, sig;

  FILE *enuc_file, *Geom_file;

  enuc_file = fopen("h2o_sto3g_enuc.dat", "r");
  double Vnn;
  fscanf(enuc_file, "%lf", &Vnn);
  fclose(enuc_file);
  printf("Nuclear repulsion energy =  %12f\n", Vnn);

  Geom_file = fopen("h2o_sto3g_geom.dat", "r");
  fscanf(Geom_file,"%zu", &Nnuc);
  printf("Number of nuclei = %u\n", Nnuc);
  arma::mat Geom(Nnuc, 4);
  for (i = 0; i < Nnuc; i++) {
    fscanf(Geom_file, "%lf %lf %lf %lf", &qi, &xi, &yi, &zi);
    Geom(i,0) = qi;
    Geom(i,1) = xi;
    Geom(i,2) = yi;
    Geom(i,3) = zi;
  }
  fclose(Geom_file);
  printf("geometry\n");
  print_arma_mat(Geom);

  arma::mat S(NBasis, NBasis);
  arma::mat T(NBasis, NBasis);
  arma::mat V(NBasis, NBasis);
  arma::mat H(NBasis, NBasis);
  arma::mat J(NBasis, NBasis);
  arma::mat K(NBasis, NBasis);
  arma::mat Mu_x_mat(NBasis, NBasis);
  arma::mat Mu_y_mat(NBasis, NBasis);
  arma::mat Mu_z_mat(NBasis, NBasis);
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

  FILE *S_file, *T_file, *V_file, *Mu_x_file, *Mu_y_file, *Mu_z_file;
  S_file = fopen("h2o_sto3g_s.dat", "r");
  T_file = fopen("h2o_sto3g_t.dat", "r");
  V_file = fopen("h2o_sto3g_v.dat", "r");
  Mu_x_file = fopen("h2o_sto3g_mux.dat", "r");
  Mu_y_file = fopen("h2o_sto3g_muy.dat", "r");
  Mu_z_file = fopen("h2o_sto3g_muz.dat", "r");

  while (fscanf(S_file, "%d %d %lf", &i, &j, &val) != EOF)
    S(i-1, j-1) = S(j-1, i-1) = val;
  while (fscanf(T_file, "%d %d %lf", &i, &j, &val) != EOF)
    T(i-1, j-1) = T(j-1, i-1) = val;
  while (fscanf(V_file, "%d %d %lf", &i, &j, &val) != EOF)
    V(i-1, j-1) = V(j-1, i-1) = val;
  while (fscanf(Mu_x_file, "%d %d %lf", &i, &j, &val) != EOF)
    Mu_x_mat(i-1, j-1) = Mu_x_mat(j-1, i-1) = val;
  while (fscanf(Mu_y_file, "%d %d %lf", &i, &j, &val) != EOF)
    Mu_y_mat(i-1, j-1) = Mu_y_mat(j-1, i-1) = val;
  while (fscanf(Mu_z_file, "%d %d %lf", &i, &j, &val) != EOF)
    Mu_z_mat(i-1, j-1) = Mu_z_mat(j-1, i-1) = val;

  fclose(S_file);
  fclose(T_file);
  fclose(V_file);
  fclose(Mu_x_file);
  fclose(Mu_y_file);
  fclose(Mu_z_file);

  arma::vec ERI = arma::vec(M, arma::fill::zeros);
  arma::vec ERI_mo = arma::vec(M, arma::fill::zeros);
  arma::vec ERI_mo2 = arma::vec(M, arma::fill::zeros);
  arma::vec ERI_mo3 = arma::vec(M, arma::fill::zeros);

  FILE *ERI_file;
  ERI_file = fopen("h2o_sto3g_eri.dat", "r");

  while (fscanf(ERI_file, "%d %d %d %d %lf", &i, &j, &k, &l, &val) != EOF) {
    mu = i-1; nu = j-1; lam = k-1; sig = l-1;
    ERI(idx4(mu,nu,lam,sig)) = val;
  }

  fclose(ERI_file);

  H = T + V;

  double thresh_E = 1.0e-10;
  double thresh_D = 1.0e-7;
  size_t iteration = 1;
  size_t max_iterations = 1024;
  double E_total, E_elec_old, E_elec_new, delta_E, rmsd_D;
  double E_kin, E_pot, E_F, E_H, E_J, E_K;

  arma::eig_sym(Lam_S_vec, L_S, S);
  Lam_S_mat.diag() = Lam_S_vec;
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
  printf("Initial Fock Matrix:\n");
  print_arma_mat(F);
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
    E_elec_new = calc_elec_energy(D_new, H, F);

    // why doesn't this work?
    // E_elec_new = calc_one_elec_prop((H + F), D_new);
    E_total = E_elec_new + Vnn;
    //do stuff and test for convergence
    delta_E = fabs(E_elec_new-E_elec_old);
    rmsd_D = rmsd_density(D_new, D_old);
    printf("%6d %20.15f %20.15f %20.15f\n", iteration, E_elec_new, rmsd_D, delta_E);
    if ( delta_E < thresh_E && rmsd_D < thresh_D){
        break;
    }
    iteration++;
  }
  build_jk(J, K, D_new, ERI);
    E_kin = calc_one_elec_prop(T, D_new);
    E_pot = calc_one_elec_prop(V, D_new);
    E_F = calc_one_elec_prop(F, D_new);
    E_H = calc_one_elec_prop(H, D_new);
    E_J = calc_one_elec_prop(J, D_new);
    E_K = calc_one_elec_prop(K, D_new);
  printf("Final Density Matrix:\n");
  print_arma_mat(D_new);
  printf("final C matrix:\n");
  print_arma_mat(C);
  printf("orbital energies:\n");
  print_arma_vec(eps_vec, NBasis);
  printf("total energy           = %20.15f\n",E_total);
  printf("kinetic energy         = %20.15f\n",E_kin);
  printf("V(e-N)                 = %20.15f\n",E_pot);
  printf("V(N-N)                 = %20.15f\n",Vnn);
  printf("E_F                    = %20.15f\n",E_F);
  printf("E_H                    = %20.15f\n",E_H);
  printf("E_J                    = %20.15f\n",E_J);
  printf("E_K                    = %20.15f\n",E_K);
  
  double mu_x, mu_y, mu_z;

  mu_x = calc_one_elec_prop(Mu_x_mat, D_new);
  mu_y = calc_one_elec_prop(Mu_y_mat, D_new);
  mu_z = calc_one_elec_prop(Mu_z_mat, D_new);
  
  for (i = 0; i < Nnuc; i++) {
    mu_x += Geom(i,0) * Geom(i,1);
    mu_y += Geom(i,0) * Geom(i,2);
    mu_z += Geom(i,0) * Geom(i,3);
  }
  printf("Mu_X =  %20.15f\n", mu_x);
  printf("Mu_Y =  %20.15f\n", mu_y);
  printf("Mu_Z =  %20.15f\n", mu_z);

  trans_ERI2(ERI, ERI_mo2, C);
  trans_ERI3(ERI, ERI_mo3, C);
  trans_ERI(ERI, ERI_mo, C);
  double mp2_e = 0.0;
  mp2_e = calc_mp2_energy(ERI_mo, eps_vec, NOcc, NBasis);
  printf("MP2 Energy =  %20.15f\n", mp2_e);
  printf("M  = %d\n",M);
  printf("M2 = %d\n",M2);
  for (int ii = 0; ii <= M2; ii++){
    printf("%5d %20.15f %20.15f %20.15f\n",ii,ERI_mo2(ii),ERI_mo(ii)-ERI_mo2(ii),ERI_mo3(ii)-ERI_mo2(ii));
  }
  /*
  int i, j, k, l, ij, kl, ijkl;
  int NBasis = 2;
  int mmax = idx4(NBasis,NBasis,NBasis,NBasis); 
  for (int ijkl= 0; ijkl < mmax+1; ijkl++) {
    idx2inv(ijkl,ij,kl);
    idx2inv(ij,i,j);
    idx2inv(kl,k,l);

    printf("%4d %4d %4d %4d %4d %4d %4d\n",ijkl,ij,kl,i,j,k,l);
  }

  for (i = 0; i < NBasis+1; i++){
    eri1=
    for (j = 0; j <= i; j++){
      for (k = 0; k <= i; k++){
        for (l = 0; l <=k && (!(i==k) || l <=j); l++){
          printf("%4d %4d %4d %4d %4d %4d %4d\n",idx4(i,j,k,l),idx2(i,j),idx2(k,l),i,j,k,l);
        }
      }
    }
  }
 */
  return 0;

}
