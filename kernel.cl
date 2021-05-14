__kernel void initializeCL( int N, int Q, double DENSITY, double LID_VELOCITY, 
                __constant double *ex, __constant double *ey, __constant int *oppos, __constant double *wt,//Q
                __global double *rho, __global double *ux, __global double *uy, __global double* sigma,//N*N 
                __global double *f, __global double *feq, __global double *f_new)//N*N*Q
{ 
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i >= N || j >= N)
        return;
    int index = i*N+j;  // column-ordering

    // initialize density and velocity fields inside the cavity

              rho[index] = DENSITY;   // density
               ux[index] = 0.0;       // x-component of velocity
               uy[index] = 0.0;       // y-component of velocity
            sigma[index] = 0.0;       // rate-of-strain field

            // specify boundary condition for the moving lid

            if(j==N-1) ux[index] = LID_VELOCITY;

            // assign initial values for distribution functions
            // along various aections using equilibriu, functions

            for(int a=0;a<Q;a++) {
        
                int index_f = a + index*Q;

                double edotu = ex[a]*ux[index] + ey[a]*uy[index];
                double udotu = ux[index]*ux[index] + uy[index]*uy[index];

                feq[index_f]   = rho[index] * wt[a] * (1.0 + 3.0*edotu + 4.5*edotu*edotu - 1.5*udotu);
                f[index_f]     = feq[index_f];
                f_new[index_f] = feq[index_f];

            }
}

__kernel void collideAndStreamCL(// READ-ONLY parameters (used by this function but not changed)
                      int N, int Q, double DENSITY, double LID_VELOCITY, double REYNOLDS_NUMBER,
                      __constant double *ex, __constant double *ey, __constant int *oppos, __constant double *wt,
                      // READ + WRITE parameters (get updated in this function)
                      __global double *rho,         // density
                      __global double *ux,         // X-velocity
                      __global double *uy,         // Y-velocity
                      __global double *sigma,      // rate-of-strain
                      __global double *f,          // distribution function
                      __global double *feq,        // equilibrium distribution function
                      __global double *f_new)      // new distribution function
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i >= N - 1 || j >= N - 1 || i == 0 || j == 0)
        return;

    // natural index
    int index = i*N + j;  // column-major ordering

    //printf("transpose index = %d\n", j*N+i);

    // calculate fluid viscosity based on the Reynolds number
    double kinematicViscosity = LID_VELOCITY * (double) N / REYNOLDS_NUMBER;

    // calculate relaxation time tau
    double tau =  0.5 + 3.0 * kinematicViscosity;

    // collision
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        double edotu = ex[a]*ux[index] + ey[a]*uy[index];
        double udotu = ux[index]*ux[index] + uy[index]*uy[index];
        feq[index_f] = rho[index] * wt[a] * (1 + 3*edotu + 4.5*edotu*edotu - 1.5*udotu);
    }

    // streaming from interior node points
    
    for(int a=0;a<Q;a++) {

        int index_f = a + index*Q;
        int index_nbr = (i+ex[a])*N + (j+ey[a]);
        int index_nbr_f = a + index_nbr * Q;
        int indexoppos = oppos[a] + index*Q;

        double tau_eff, tau_t, C_Smagorinsky;  // turbulence model parameters

        C_Smagorinsky = 0.16;

        // tau_t = additional contribution to the relaxation time 
        //         because of the "eddy viscosity" model
        // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        // REFERENCE: Krafczyk M., Tolke J. and Luo L.-S. (2003)
        //            Large-Eddy Simulations with a Multiple-Relaxation-Time LBE Model
        //            International Journal of Modern Physics B, Vol.17, 33-39
        // =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        tau_t = 0.5*(pow(pow(tau,2) + 18.0*pow(C_Smagorinsky,2)*sigma[index],0.5) - tau);

        // the effective relaxation time accounts for the additional "eddy viscosity"
        // effects. Note that tau_eff now varies from point to point in the domain, and is
        // larger for large strain rates. If the strain rate is zero, tau_eff = 0 and we
        // revert back to the original (laminar) LBM scheme where tau_eff = tau.

        tau_eff = tau + tau_t;

        // post-collision distribution at (i,j) along "a"
        double f_plus = f[index_f] - (f[index_f] - feq[index_f])/tau_eff;

        int iS = i + ex[a]; int jS = j + ey[a];

        if((iS==0) || (iS==N-1) || (jS==0) || (jS==N-1) ) {
            // bounce back
            double ubdote = ux[index_nbr]*ex[a] + uy[index_nbr]*ey[a];
            f_new[indexoppos] = f_plus - 6.0 * DENSITY * wt[a] * ubdote;
        }
        else {
            // stream to neighbor
            f_new[index_nbr_f] = f_plus;
        }
    }
}

__kernel void macroVarCL( // READ-ONLY parameters (used by this function but not changed)
               int N, int Q, double DENSITY, double LID_VELOCITY, double REYNOLDS_NUMBER,
               __constant double *ex, __constant double *ey, __constant int *oppos, __constant double *wt,
               // READ + WRITE parameters (get updated in this function)
               __global double *rho,         // density
               __global double *ux,         // X-velocity
               __global double *uy,         // Y-velocity
               __global double *sigma,      // rate-of-strain
               __global double *f,          // distribution function
               __global double *feq,        // equilibrium distribution function
               __global double *f_new)      // new distribution function
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if(i >= N - 1 || j >= N - 1 || i == 0 || j == 0)
        return;

    // natural index
    int index = i*N + j;  // column-major ordering

    // push f_new into f
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        f[index_f] = f_new[index_f];
    }

    // update density at interior nodes
    rho[index]=0.0;
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        rho[index] += f_new[index_f];
    }

    // update velocity at interior nodes
    double velx=0.0;
    double vely=0.0;
    for(int a=0;a<Q;a++) {
        int index_f = a + index*Q;
        velx += f_new[index_f]*ex[a];
        vely += f_new[index_f]*ey[a];
    }
    ux[index] = velx/rho[index];
    uy[index] = vely/rho[index];

    // update the rate-of-strain field
    double sum_xx = 0.0, sum_xy = 0.0, sum_xz = 0.0;
    double sum_yx = 0.0, sum_yy = 0.0, sum_yz = 0.0;
    double sum_zx = 0.0, sum_zy = 0.0, sum_zz = 0.0;
    for(int a=1; a<Q; a++)
    {
        int index_f = a + index*Q;

        sum_xx = sum_xx + (f_new[index_f] - feq[index_f])*ex[a]*ex[a];
        sum_xy = sum_xy + (f_new[index_f] - feq[index_f])*ex[a]*ey[a];
        sum_xz = 0.0;
        sum_yx = sum_xy;
        sum_yy = sum_yy + (f_new[index_f] - feq[index_f])*ey[a]*ey[a];
        sum_yz = 0.0;
        sum_zx = 0.0;
        sum_zy = 0.0;
        sum_zz = 0.0;
    }

    // evaluate |S| (magnitude of the strain-rate)
    sigma[index] = pow(sum_xx,2) + pow(sum_xy,2) + pow(sum_xz,2)
                 + pow(sum_yx,2) + pow(sum_yy,2) + pow(sum_yz,2)
                 + pow(sum_zx,2) + pow(sum_zy,2) + pow(sum_zz,2);

    sigma[index] = pow(sigma[index],0.5);

}
