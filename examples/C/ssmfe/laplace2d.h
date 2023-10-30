#pragma once

/* Multiply x by Laplacian matrix for a square of size mx by my */
void apply_laplacian(
      int mx, int my, int nx, const double *x, double *Ax
      ) {
   for(int k=0; k<nx; k++) {
      for(int i=0; i<mx; i++) {
         for(int j=0; j<my; j++) {
            double z = 4*x[k*mx*my + j*mx + i];
            if( i > 0    ) z -= x[k*mx*my + j*mx + i-1];
            if( j > 0    ) z -= x[k*mx*my + (j-1)*mx + i];
            if( i < mx-1 ) z -= x[k*mx*my + j*mx + i+1];
            if( j < my-1 ) z -= x[k*mx*my + (j+1)*mx + i];
            Ax[k*mx*my + j*mx + i] = z;
         }
      }
   }
}

/* Laplacian matrix for a rectangle of size nx x ny */
void set_laplacian_matrix(
      int nx, int ny, int lda, double *a
      ) {
   for(int j=0; j<nx*ny; j++)
   for(int i=0; i<lda; i++)
      a[j*lda + i] = 0.0;
   for(int ix=0; ix<nx; ix++) {
      for(int iy=0; iy<ny; iy++) {
        int i = ix + iy*nx;
        a[i*lda + i] = 4;
        if( ix >  0   ) a[(i-1)*lda + i] = -1;
        if( ix < nx-1 ) a[(i+1)*lda + i] = -1;
        if( iy > 0    ) a[(i-nx)*lda + i] = -1;
        if( iy < ny-1 ) a[(i+nx)*lda + i] = -1;
      }
   }
}

/* Apply one Gauss-Seidel step for preconditioning */
void apply_gauss_seidel_step(
      int mx, int my, int nx, const double *x, double *Tx
      ) {
   for(int i=0; i<mx; i++)
      for(int j=0; j<my; j++)
         for(int k=0; k<nx; k++)
            Tx[k*mx*my + j*mx + i] = x[k*mx*my + j*mx + i]/4;
   for(int k=0; k<nx; k++) {
      /* forward update */
      for(int i=0; i<mx; i++) {
         for(int j=0; j<my; j++) {
            double z = 0.0;
            if( i > 0    ) z += Tx[k*mx*my + j*mx + i-1];
            if( j > 0    ) z += Tx[k*mx*my + (j-1)*mx + i];
            if( i < mx-1 ) z += Tx[k*mx*my + j*mx + i+1];
            if( j < my-1 ) z += Tx[k*mx*my + (j+1)*mx + i];
            Tx[k*mx*my + j*mx + i] = (Tx[k*mx*my + j*mx + i] + z/4)/4;
         }
      }
      /* backward update */
      for(int i=mx-1; i>=0; i--) {
         for(int j=my-1; j>=0; j--) {
            double z = 0.0;
            if( i > 0    ) z += Tx[k*mx*my + j*mx + i-1];
            if( j > 0    ) z += Tx[k*mx*my + (j-1)*mx + i];
            if( i < mx-1 ) z += Tx[k*mx*my + j*mx + i+1];
            if( j < my-1 ) z += Tx[k*mx*my + (j+1)*mx + i];
            Tx[k*mx*my + j*mx + i] = (Tx[k*mx*my + j*mx + i] + z/4)/4;
         }
      }
   }
}
