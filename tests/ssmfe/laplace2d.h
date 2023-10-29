#pragma once

/* Multiply x by Laplacian matrix for a square of size mx by my */
void apply_laplacian(
      int mx, int my, int nx, const double *x, double *Ax
      ) {
   for(int k=0; k<nx; k++) {
      for(int i=0; i<mx; i++) {
         for(int j=0; j<my; j++) {
            double z = 4*x[k*my*mx + j*mx + i];
            if( i > 0    ) z -= x[k*my*mx + j*mx + i-1];
            if( j > 0    ) z -= x[k*my*mx + (j-1)*mx + i];
            if( i < mx-1 ) z -= x[k*my*mx + j*mx + i+1];
            if( j < my-1 ) z -= x[k*my*mx + (j+1)*mx + i];
            Ax[k*my*mx + j*mx + i] = z;
         }
      }
   }
}

void apply_laplacian_z(
      int mx, int my, int nx, 
      const double complex *x, double complex *Ax
      ) {
   for(int k=0; k<nx; k++) {
      for(int i=0; i<mx; i++) {
         for(int j=0; j<my; j++) {
            double complex z = 4*x[k*my*mx + j*mx + i];
            if( i > 0    ) z -= x[k*my*mx + j*mx + i-1];
            if( j > 0    ) z -= x[k*my*mx + (j-1)*mx + i];
            if( i < mx-1 ) z -= x[k*my*mx + j*mx + i+1];
            if( j < my-1 ) z -= x[k*my*mx + (j+1)*mx + i];
            Ax[k*my*mx + j*mx + i] = z;
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

void set_laplacian_matrix_z(
      int nx, int ny, int lda, double complex *a
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
            Tx[k*my*mx + j*mx + i] = x[k*my*mx + j*mx + i]/4;
   for(int k=0; k<nx; k++) {
      /* forward update */
      for(int i=0; i<mx; i++) {
         for(int j=0; j<my; j++) {
            double z = 0.0;
            if( i > 0    ) z += Tx[k*my*mx + j*mx + i-1];
            if( j > 0    ) z += Tx[k*my*mx + (j-1)*mx + i];
            if( i < mx-1 ) z += Tx[k*my*mx + j*mx + i+1];
            if( j < my-1 ) z += Tx[k*my*mx + (j+1)*mx + i];
            Tx[k*my*mx + j*mx + i] = (Tx[k*my*mx + j*mx + i] + z/4)/4;
         }
      }
      /* backward update */
      for(int i=mx-1; i>=0; i--) {
         for(int j=my-1; j>=0; j--) {
            double z = 0.0;
            if( i > 0    ) z += Tx[k*my*mx + j*mx + i-1];
            if( j > 0    ) z += Tx[k*my*mx + (j-1)*mx + i];
            if( i < mx-1 ) z += Tx[k*my*mx + j*mx + i+1];
            if( j < my-1 ) z += Tx[k*my*mx + (j+1)*mx + i];
            Tx[k*my*mx + j*mx + i] = (Tx[k*my*mx + j*mx + i] + z/4)/4;
         }
      }
   }
}

void apply_gauss_seidel_step_z(
      int mx, int my, int nx, 
      const double complex *x, double complex *Tx
      ) {
   for(int i=0; i<mx; i++)
      for(int j=0; j<my; j++)
         for(int k=0; k<nx; k++)
            Tx[k*my*mx + j*mx + i] = x[k*my*mx + j*mx + i]/4;
   for(int k=0; k<nx; k++) {
      /* forward update */
      for(int i=0; i<mx; i++) {
         for(int j=0; j<my; j++) {
            double z = 0.0;
            if( i > 0    ) z += Tx[k*my*mx + j*mx + i-1];
            if( j > 0    ) z += Tx[k*my*mx + (j-1)*mx + i];
            if( i < mx-1 ) z += Tx[k*my*mx + j*mx + i+1];
            if( j < my-1 ) z += Tx[k*my*mx + (j+1)*mx + i];
            Tx[k*my*mx + j*mx + i] = (Tx[k*my*mx + j*mx + i] + z/4)/4;
         }
      }
      /* backward update */
      for(int i=mx-1; i>=0; i--) {
         for(int j=my-1; j>=0; j--) {
            double z = 0.0;
            if( i > 0    ) z += Tx[k*my*mx + j*mx + i-1];
            if( j > 0    ) z += Tx[k*my*mx + (j-1)*mx + i];
            if( i < mx-1 ) z += Tx[k*my*mx + j*mx + i+1];
            if( j < my-1 ) z += Tx[k*my*mx + (j+1)*mx + i];
            Tx[k*my*mx + j*mx + i] = (Tx[k*my*mx + j*mx + i] + z/4)/4;
         }
      }
   }
}

