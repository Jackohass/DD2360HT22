#include "Particles.h"
#include "Alloc.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 1024

//@@ Insert code to implement timer start
double currTime;

double startTime() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  currTime = ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
  return currTime;
}

//@@ Insert code to implement timer stop
double stopTime() {
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6) - currTime;
}


/** allocate particle arrays */
void particle_allocate(struct parameters* param, struct particles* part, int is)
{
    
    // set species ID
    part->species_ID = is;
    // number of particles
    part->nop = param->np[is];
    // maximum number of particles
    part->npmax = param->npMax[is];
    
    // choose a different number of mover iterations for ions and electrons
    if (param->qom[is] < 0){  //electrons
        part->NiterMover = param->NiterMover;
        part->n_sub_cycles = param->n_sub_cycles;
    } else {                  // ions: only one iteration
        part->NiterMover = 1;
        part->n_sub_cycles = 1;
    }
    
    // particles per cell
    part->npcelx = param->npcelx[is];
    part->npcely = param->npcely[is];
    part->npcelz = param->npcelz[is];
    part->npcel = part->npcelx*part->npcely*part->npcelz;
    
    // cast it to required precision
    part->qom = (FPpart) param->qom[is];
    
    long npmax = part->npmax;
    
    // initialize drift and thermal velocities
    // drift
    part->u0 = (FPpart) param->u0[is];
    part->v0 = (FPpart) param->v0[is];
    part->w0 = (FPpart) param->w0[is];
    // thermal
    part->uth = (FPpart) param->uth[is];
    part->vth = (FPpart) param->vth[is];
    part->wth = (FPpart) param->wth[is];
    
    
    //////////////////////////////
    /// ALLOCATION PARTICLE ARRAYS
    //////////////////////////////
    part->x = new FPpart[npmax];
    part->y = new FPpart[npmax];
    part->z = new FPpart[npmax];
    // allocate velocity
    part->u = new FPpart[npmax];
    part->v = new FPpart[npmax];
    part->w = new FPpart[npmax];
    // allocate charge = q * statistical weight
    part->q = new FPinterp[npmax];
    
}
/** deallocate */
void particle_deallocate(struct particles* part)
{
    // deallocate particle variables
    delete[] part->x;
    delete[] part->y;
    delete[] part->z;
    delete[] part->u;
    delete[] part->v;
    delete[] part->w;
    delete[] part->q;
}

/** particle mover */
int mover_PC_cpu(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    // auxiliary variables
    FPpart dt_sub_cycling = (FPpart) param->dt/((double) part->n_sub_cycles);
    FPpart dto2 = .5*dt_sub_cycling, qomdt2 = part->qom*dto2/param->c;
    FPpart omdtsq, denom, ut, vt, wt, udotb;
    
    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;
    
    // interpolation densities
    int ix,iy,iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];
    
    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;
    
    startTime();
    // start subcycling
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        // move each particle with new fields
        for (int i=0; i <  part->nop; i++){
            xptilde = part->x[i];
            yptilde = part->y[i];
            zptilde = part->z[i];
            // calculate the average velocity iteratively
            for(int innter=0; innter < part->NiterMover; innter++){
                // interpolation G-->P
                ix = 2 +  int((part->x[i] - grd->xStart)*grd->invdx);
                iy = 2 +  int((part->y[i] - grd->yStart)*grd->invdy);
                iz = 2 +  int((part->z[i] - grd->zStart)*grd->invdz);
                
                // calculate weights
                xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
                eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
                zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
                xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
                eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
                zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
                for (int ii = 0; ii < 2; ii++)
                    for (int jj = 0; jj < 2; jj++)
                        for (int kk = 0; kk < 2; kk++)
                            weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
                
                // set to zero local electric and magnetic field
                Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
                
                for (int ii=0; ii < 2; ii++)
                    for (int jj=0; jj < 2; jj++)
                        for(int kk=0; kk < 2; kk++){
                            Exl += weight[ii][jj][kk]*field->Ex[ix- ii][iy -jj][iz- kk ];
                            Eyl += weight[ii][jj][kk]*field->Ey[ix- ii][iy -jj][iz- kk ];
                            Ezl += weight[ii][jj][kk]*field->Ez[ix- ii][iy -jj][iz -kk ];
                            Bxl += weight[ii][jj][kk]*field->Bxn[ix- ii][iy -jj][iz -kk ];
                            Byl += weight[ii][jj][kk]*field->Byn[ix- ii][iy -jj][iz -kk ];
                            Bzl += weight[ii][jj][kk]*field->Bzn[ix- ii][iy -jj][iz -kk ];
                        }
                
                // end interpolation
                omdtsq = qomdt2*qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
                denom = 1.0/(1.0 + omdtsq);
                // solve the position equation
                ut= part->u[i] + qomdt2*Exl;
                vt= part->v[i] + qomdt2*Eyl;
                wt= part->w[i] + qomdt2*Ezl;
                udotb = ut*Bxl + vt*Byl + wt*Bzl;
                // solve the velocity equation
                uptilde = (ut+qomdt2*(vt*Bzl -wt*Byl + qomdt2*udotb*Bxl))*denom;
                vptilde = (vt+qomdt2*(wt*Bxl -ut*Bzl + qomdt2*udotb*Byl))*denom;
                wptilde = (wt+qomdt2*(ut*Byl -vt*Bxl + qomdt2*udotb*Bzl))*denom;
                // update position
                part->x[i] = xptilde + uptilde*dto2;
                part->y[i] = yptilde + vptilde*dto2;
                part->z[i] = zptilde + wptilde*dto2;
                
                
            } // end of iteration
            // update the final position and velocity
            part->u[i]= 2.0*uptilde - part->u[i];
            part->v[i]= 2.0*vptilde - part->v[i];
            part->w[i]= 2.0*wptilde - part->w[i];
            part->x[i] = xptilde + uptilde*dt_sub_cycling;
            part->y[i] = yptilde + vptilde*dt_sub_cycling;
            part->z[i] = zptilde + wptilde*dt_sub_cycling;
            
            
            //////////
            //////////
            ////////// BC
                                        
            // X-DIRECTION: BC particles
            if (part->x[i] > grd->Lx){
                if (param->PERIODICX==true){ // PERIODIC
                    part->x[i] = part->x[i] - grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = 2*grd->Lx - part->x[i];
                }
            }
                                                                        
            if (part->x[i] < 0){
                if (param->PERIODICX==true){ // PERIODIC
                   part->x[i] = part->x[i] + grd->Lx;
                } else { // REFLECTING BC
                    part->u[i] = -part->u[i];
                    part->x[i] = -part->x[i];
                }
            }
                
            
            // Y-DIRECTION: BC particles
            if (part->y[i] > grd->Ly){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] - grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = 2*grd->Ly - part->y[i];
                }
            }
                                                                        
            if (part->y[i] < 0){
                if (param->PERIODICY==true){ // PERIODIC
                    part->y[i] = part->y[i] + grd->Ly;
                } else { // REFLECTING BC
                    part->v[i] = -part->v[i];
                    part->y[i] = -part->y[i];
                }
            }
                                                                        
            // Z-DIRECTION: BC particles
            if (part->z[i] > grd->Lz){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] - grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = 2*grd->Lz - part->z[i];
                }
            }
                                                                        
            if (part->z[i] < 0){
                if (param->PERIODICZ==true){ // PERIODIC
                    part->z[i] = part->z[i] + grd->Lz;
                } else { // REFLECTING BC
                    part->w[i] = -part->w[i];
                    part->z[i] = -part->z[i];
                }
            }
                                                                        
            
            
        }  // end of subcycling
    } // end of one particle
    double CPUTime = stopTime()*1000;
    printf("CPU: %f ms\n", CPUTime);
                                                                        
    return(0); // exit succcesfully
} // end of the mover

#define XSTRIDE 0
#define YSTRIDE 1
#define ZSTRIDE 2
#define USTRIDE 3
#define VSTRIDE 4
#define WSTRIDE 5

struct GPUParam {
    long nop;
    int NiterMover;
    int n_sub_cycles;
    long npmax;
    FPpart dt_sub_cycling;
    FPpart dto2;
    FPpart qomdt2;
    bool xPerd, yPerd,  zPerd;
    double grdxStart, grdyStart, grdzStart;
    double grdinvdx,  grdinvdy, grdinvdz;
    int nyn, nzn;
    double Lx, Ly, Lz;
    FPfield invVOL;
};

// move each particle with new fields
__global__ void moveParticles(struct GPUParam p,
    FPpart* outX, FPpart* outY, FPpart* outZ, FPpart* outU, FPpart* outV, FPpart* outW,
    FPfield* XN, FPfield* YN, FPfield* ZN,
    FPfield* Ex, FPfield* Ey, FPfield* Ez,
    FPfield* Bxn, FPfield* Byn, FPfield* Bzn)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= p.nop) return;

    // auxiliary variables
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix, iy, iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde;

    FPpart xTemp = outX[idx];
    FPpart yTemp = outY[idx];
    FPpart zTemp = outZ[idx];
    FPpart uTemp = outU[idx];
    FPpart vTemp = outV[idx];
    FPpart wTemp = outW[idx];

    xptilde = xTemp;
    yptilde = yTemp;
    zptilde = zTemp;
    // calculate the average velocity iteratively
    for(int innter=0; innter < p.NiterMover; innter++){
        // interpolation G-->P
        ix = 2 +  int((xTemp - p.grdxStart)*p.grdinvdx);
        iy = 2 +  int((yTemp - p.grdyStart)*p.grdinvdy);
        iz = 2 +  int((zTemp - p.grdzStart)*p.grdinvdz);
        
        // calculate weights
        xi[0]   = xTemp - XN[get_idx(ix - 1, iy, iz, p.nyn, p.nzn)];
        eta[0]  = yTemp - YN[get_idx(ix, iy - 1, iz, p.nyn, p.nzn)];
        zeta[0] = zTemp - ZN[get_idx(ix, iy, iz - 1, p.nyn, p.nzn)];
        int idxXEZ = get_idx(ix, iy, iz, p.nyn, p.nzn);
        xi[1]   = XN[idxXEZ] - xTemp;
        eta[1]  = YN[idxXEZ] - yTemp;
        zeta[1] = ZN[idxXEZ] - zTemp;
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * p.invVOL;
        
        // set to zero local electric and magnetic field
        Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
        
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++){
                    int idxE = get_idx(ix- ii, iy -jj, iz -kk, p.nyn, p.nzn);
                    Exl += weight[ii][jj][kk]*Ex[idxE];
                    Eyl += weight[ii][jj][kk]*Ey[idxE];
                    Ezl += weight[ii][jj][kk]*Ez[idxE];
                    Bxl += weight[ii][jj][kk]*Bxn[idxE];
                    Byl += weight[ii][jj][kk]*Byn[idxE];
                    Bzl += weight[ii][jj][kk]*Bzn[idxE];
                }
        
        // end interpolation
        omdtsq = p.qomdt2*p.qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
        denom = 1.0/(1.0 + omdtsq);
        // solve the position equation
        ut= uTemp + p.qomdt2*Exl;
        vt= vTemp + p.qomdt2*Eyl;
        wt= wTemp + p.qomdt2*Ezl;
        udotb = ut*Bxl + vt*Byl + wt*Bzl;
        // solve the velocity equation
        uptilde = (ut+p.qomdt2*(vt*Bzl -wt*Byl + p.qomdt2*udotb*Bxl))*denom;
        vptilde = (vt+p.qomdt2*(wt*Bxl -ut*Bzl + p.qomdt2*udotb*Byl))*denom;
        wptilde = (wt+p.qomdt2*(ut*Byl -vt*Bxl + p.qomdt2*udotb*Bzl))*denom;
        // update position
        xTemp = xptilde + uptilde*p.dto2;
        yTemp = yptilde + vptilde*p.dto2;
        zTemp = zptilde + wptilde*p.dto2;
        
    } // end of iteration

    // update the final position and velocity
    uTemp= 2.0*uptilde - uTemp;
    vTemp= 2.0*vptilde - vTemp;
    wTemp= 2.0*wptilde - wTemp;
    xTemp = xptilde + uptilde*p.dt_sub_cycling;
    yTemp = yptilde + vptilde*p.dt_sub_cycling;
    zTemp = zptilde + wptilde*p.dt_sub_cycling;
    
    
    //////////
    //////////
    ////////// BC
                                
    // X-DIRECTION: BC particles
    if (xTemp > p.Lx){
        if (p.xPerd==true){ // PERIODIC
            xTemp = xTemp - p.Lx;
        } else { // REFLECTING BC
            uTemp = -uTemp;
            xTemp = 2*p.Lx - xTemp;
        }
    }
                                                                
    if (xTemp < 0){
        if (p.xPerd==true){ // PERIODIC
            xTemp = xTemp + p.Lx;
        } else { // REFLECTING BC
            uTemp = -uTemp;
            xTemp = -xTemp;
        }
    }
        
    
    // Y-DIRECTION: BC particles
    if (yTemp > p.Ly){
        if (p.yPerd==true){ // PERIODIC
            yTemp = yTemp - p.Ly;
        } else { // REFLECTING BC
            vTemp = -vTemp;
            yTemp = 2*p.Ly - yTemp;
        }
    }
                                                                
    if (yTemp < 0){
        if (p.yPerd==true){ // PERIODIC
            yTemp = yTemp + p.Ly;
        } else { // REFLECTING BC
            vTemp = -vTemp;
            yTemp = -yTemp;
        }
    }
                                                                
    // Z-DIRECTION: BC particles
    if (zTemp > p.Lz){
        if (p.zPerd==true){ // PERIODIC
            zTemp = zTemp - p.Lz;
        } else { // REFLECTING BC
            wTemp = -wTemp;
            zTemp = 2*p.Lz - zTemp;
        }
    }
                                                                
    if (zTemp < 0){
        if (p.zPerd==true){ // PERIODIC
            zTemp = zTemp + p.Lz;
        } else { // REFLECTING BC
            wTemp = -wTemp;
            zTemp = -zTemp;
        }
    }

    outU[idx] = uTemp;
    outV[idx] = vTemp;
    outW[idx] = wTemp;
    outX[idx] = xTemp;
    outY[idx] = yTemp;
    outZ[idx] = zTemp;
}

// move each particle with new fields
/*__global__ void moveParticles(struct GPUParam p, FPpart* outP,
    FPfield* XN, FPfield* YN, FPfield* ZN,
    FPfield* Ex, FPfield* Ey, FPfield* Ez,
    FPfield* Bxn, FPfield* Byn, FPfield* Bzn)
{
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= p.nop) return;

    // auxiliary variables
    FPpart omdtsq, denom, ut, vt, wt, udotb;

    // local (to the particle) electric and magnetic field
    FPfield Exl=0.0, Eyl=0.0, Ezl=0.0, Bxl=0.0, Byl=0.0, Bzl=0.0;

    // interpolation densities
    int ix, iy, iz;
    FPfield weight[2][2][2];
    FPfield xi[2], eta[2], zeta[2];

    // intermediate particle position and velocity
    FPpart xptilde, yptilde, zptilde, uptilde, vptilde, wptilde, uTemp, vTemp, wTemp;

    FPpart xTemp = outP[idx + p.npmax*XSTRIDE];
    FPpart yTemp = outP[idx + p.npmax*YSTRIDE];
    FPpart zTemp = outP[idx + p.npmax*ZSTRIDE];

    // start subcycling
    for (int i_sub=0; i_sub < p.n_sub_cycles; i_sub++)
    {
        xptilde = xTemp;
        yptilde = yTemp;
        zptilde = zTemp;
        
        // calculate the average velocity iteratively
        for(int innter=0; innter < p.NiterMover; innter++){
            // interpolation G-->P
            ix = 2 +  int((xTemp - p.grdxStart)*p.grdinvdx);
            iy = 2 +  int((yTemp - p.grdyStart)*p.grdinvdy);
            iz = 2 +  int((zTemp - p.grdzStart)*p.grdinvdz);
            
            // calculate weights
            xi[0]   = xTemp - XN[get_idx(ix - 1, iy, iz, p.nyn, p.nzn)];
            eta[0]  = yTemp - YN[get_idx(ix, iy - 1, iz, p.nyn, p.nzn)];
            zeta[0] = zTemp - ZN[get_idx(ix, iy, iz - 1, p.nyn, p.nzn)];
            int idxXEZ = get_idx(ix, iy, iz, p.nyn, p.nzn);
            xi[1]   = XN[idxXEZ] - xTemp;
            eta[1]  = YN[idxXEZ] - yTemp;
            zeta[1] = ZN[idxXEZ] - zTemp;
            for (int ii = 0; ii < 2; ii++)
                for (int jj = 0; jj < 2; jj++)
                    for (int kk = 0; kk < 2; kk++)
                        weight[ii][jj][kk] = xi[ii] * eta[jj] * zeta[kk] * p.invVOL;
            
            // set to zero local electric and magnetic field
            Exl=0.0, Eyl = 0.0, Ezl = 0.0, Bxl = 0.0, Byl = 0.0, Bzl = 0.0;
            
            for (int ii=0; ii < 2; ii++)
                for (int jj=0; jj < 2; jj++)
                    for(int kk=0; kk < 2; kk++){
                        int idxE = get_idx(ix- ii, iy -jj, iz -kk, p.nyn, p.nzn);
                        Exl += weight[ii][jj][kk]*Ex[idxE];
                        Eyl += weight[ii][jj][kk]*Ey[idxE];
                        Ezl += weight[ii][jj][kk]*Ez[idxE];
                        Bxl += weight[ii][jj][kk]*Bxn[idxE];
                        Byl += weight[ii][jj][kk]*Byn[idxE];
                        Bzl += weight[ii][jj][kk]*Bzn[idxE];
                    }
            
            // end interpolation
            omdtsq = p.qomdt2*p.qomdt2*(Bxl*Bxl+Byl*Byl+Bzl*Bzl);
            denom = 1.0/(1.0 + omdtsq);
            // solve the position equation
            ut= uTemp + p.qomdt2*Exl;
            vt= vTemp + p.qomdt2*Eyl;
            wt= wTemp + p.qomdt2*Ezl;
            udotb = ut*Bxl + vt*Byl + wt*Bzl;
            // solve the velocity equation
            uptilde = (ut+p.qomdt2*(vt*Bzl -wt*Byl + p.qomdt2*udotb*Bxl))*denom;
            vptilde = (vt+p.qomdt2*(wt*Bxl -ut*Bzl + p.qomdt2*udotb*Byl))*denom;
            wptilde = (wt+p.qomdt2*(ut*Byl -vt*Bxl + p.qomdt2*udotb*Bzl))*denom;
            // update position
            xTemp = xptilde + uptilde*p.dto2;
            yTemp = yptilde + vptilde*p.dto2;
            zTemp = zptilde + wptilde*p.dto2;
            
        } // end of iteration

        // update the final position and velocity
        uTemp= 2.0*uptilde - uTemp;
        vTemp= 2.0*vptilde - vTemp;
        wTemp= 2.0*wptilde - wTemp;
        xTemp = xptilde + uptilde*p.dt_sub_cycling;
        yTemp = yptilde + vptilde*p.dt_sub_cycling;
        zTemp = zptilde + wptilde*p.dt_sub_cycling;
        
        
        //////////
        //////////
        ////////// BC
                                    
        // X-DIRECTION: BC particles
        if (xTemp > p.Lx){
            if (p.xPerd==true){ // PERIODIC
                xTemp = xTemp - p.Lx;
            } else { // REFLECTING BC
                uTemp = -uTemp;
                xTemp = 2*p.Lx - xTemp;
            }
        }
                                                                    
        if (xTemp < 0){
            if (p.xPerd==true){ // PERIODIC
                xTemp = xTemp + p.Lx;
            } else { // REFLECTING BC
                uTemp = -uTemp;
                xTemp = -xTemp;
            }
        }
            
        
        // Y-DIRECTION: BC particles
        if (yTemp > p.Ly){
            if (p.yPerd==true){ // PERIODIC
                yTemp = yTemp - p.Ly;
            } else { // REFLECTING BC
                vTemp = -vTemp;
                yTemp = 2*p.Ly - yTemp;
            }
        }
                                                                    
        if (yTemp < 0){
            if (p.yPerd==true){ // PERIODIC
                yTemp = yTemp + p.Ly;
            } else { // REFLECTING BC
                vTemp = -vTemp;
                yTemp = -yTemp;
            }
        }
                                                                    
        // Z-DIRECTION: BC particles
        if (zTemp > p.Lz){
            if (p.zPerd==true){ // PERIODIC
                zTemp = zTemp - p.Lz;
            } else { // REFLECTING BC
                wTemp = -wTemp;
                zTemp = 2*p.Lz - zTemp;
            }
        }
                                                                    
        if (zTemp < 0){
            if (p.zPerd==true){ // PERIODIC
                zTemp = zTemp + p.Lz;
            } else { // REFLECTING BC
                wTemp = -wTemp;
                zTemp = -zTemp;
            }
        }
    } // end of one particle

    outP[idx + p.npmax*USTRIDE] = uTemp;
    outP[idx + p.npmax*VSTRIDE] = vTemp;
    outP[idx + p.npmax*WSTRIDE] = wTemp;
    outP[idx + p.npmax*XSTRIDE] = xTemp;
    outP[idx + p.npmax*YSTRIDE] = yTemp;
    outP[idx + p.npmax*ZSTRIDE] = zTemp;
}*/

/** GPU particle mover */
int mover_PC(struct particles* part, struct EMfield* field, struct grid* grd, struct parameters* param)
{
    // print species and subcycling
    std::cout << "***  MOVER with SUBCYCLYING "<< param->n_sub_cycles << " - species " << part->species_ID << " ***" << std::endl;
 
    //Struct for structured GPU parameters
    struct GPUParam paramsG;

    //GPU memory variables
    //FPpart* deviceOutput;
    FPpart* devicePartX;
    FPpart* devicePartY;
    FPpart* devicePartZ;
    FPpart* devicePartU;
    FPpart* devicePartV;
    FPpart* devicePartW;

    FPfield* deviceXN;
    FPfield* deviceYN;
    FPfield* deviceZN;

    FPfield* deviceEx;
    FPfield* deviceEy;
    FPfield* deviceEz;

    FPfield* deviceBxn;
    FPfield* deviceByn;
    FPfield* deviceBzn;

    
    // auxiliary variables
    paramsG.dt_sub_cycling  = (FPpart) param->dt/((double) part->n_sub_cycles);
    paramsG.dto2 = .5*paramsG.dt_sub_cycling;
    paramsG.qomdt2 = part->qom*paramsG.dto2/param->c;

    //Set of the rest of the GPU parameters.
    paramsG.nop = part->nop;
    paramsG.grdxStart = grd->xStart;
    paramsG.grdyStart = grd->yStart;
    paramsG.grdzStart = grd->zStart;
    paramsG.grdinvdx = grd->invdx;
    paramsG.grdinvdy = grd->invdy;
    paramsG.grdinvdz = grd->invdz;
    paramsG.NiterMover = part->NiterMover;
    paramsG.n_sub_cycles = part->n_sub_cycles;
    paramsG.npmax = part->npmax;
    paramsG.xPerd = param->PERIODICX;
    paramsG.yPerd = param->PERIODICY;
    paramsG.zPerd = param->PERIODICZ;
    paramsG.nyn = grd->nyn;
    paramsG.nzn = grd->nzn;
    paramsG.Lx = grd->Lx;
    paramsG.Ly = grd->Ly;
    paramsG.Lz = grd->Lz;
    paramsG.invVOL = grd->invVOL;

    //Set size
    size_t particleSize = sizeof(FPpart)*paramsG.npmax;
    size_t nodeSize = sizeof(FPfield)*grd->nxn*grd->nyn*grd->nzn;

    //Initalize Cuda
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device : %s\n", prop.name);
    cudaSetDevice(0);

    //Allocate GPU memory here
    cudaMalloc(&devicePartX, particleSize);
    cudaMalloc(&devicePartY, particleSize);
    cudaMalloc(&devicePartZ, particleSize);
    cudaMalloc(&devicePartU, particleSize);
    cudaMalloc(&devicePartV, particleSize);
    cudaMalloc(&devicePartW, particleSize);

    cudaMalloc(&deviceXN, nodeSize);
    cudaMalloc(&deviceYN, nodeSize);
    cudaMalloc(&deviceZN, nodeSize);

    cudaMalloc(&deviceEx, nodeSize);
    cudaMalloc(&deviceEy, nodeSize);
    cudaMalloc(&deviceEz, nodeSize);
    
    cudaMalloc(&deviceBxn, nodeSize);
    cudaMalloc(&deviceByn, nodeSize);
    cudaMalloc(&deviceBzn, nodeSize);

    //Copy memory to the GPU here
    startTime();
    cudaMemcpy(devicePartX, part->x, particleSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePartY, part->y, particleSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePartZ, part->z, particleSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePartU, part->u, particleSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePartV, part->v, particleSize, cudaMemcpyHostToDevice);
    cudaMemcpy(devicePartW, part->w, particleSize, cudaMemcpyHostToDevice);

    cudaMemcpy(deviceXN, grd->XN_flat, nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceYN, grd->YN_flat, nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceZN, grd->ZN_flat, nodeSize, cudaMemcpyHostToDevice);

    cudaMemcpy(deviceEx, field->Ex_flat, nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEy, field->Ey_flat, nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceEz, field->Ez_flat, nodeSize, cudaMemcpyHostToDevice);

    cudaMemcpy(deviceBxn, field->Bxn_flat, nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceByn, field->Byn_flat, nodeSize, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceBzn, field->Bzn_flat, nodeSize, cudaMemcpyHostToDevice);
    double HToDTime = stopTime()*1000;
    
    //@@ Initialize the grid and block dimensions here
    dim3 block(BLOCK_SIZE);
    dim3 grid((int)ceil(((double)paramsG.nop)/((double)BLOCK_SIZE)));

    //@@ Launch the GPU Kernel here
    startTime();
    for (int i_sub=0; i_sub <  part->n_sub_cycles; i_sub++){
        moveParticles<<<grid, block>>>(paramsG, 
            devicePartX, devicePartY, devicePartZ, devicePartU, devicePartV, devicePartW,
            deviceXN, deviceYN, deviceZN,
            deviceEx, deviceEy, deviceEz,
            deviceBxn, deviceByn, deviceBzn);
        cudaDeviceSynchronize();
    }
    double kernelTime = stopTime()*1000;
    
    //@@ Copy the GPU memory back to the CPU here
    startTime();
    cudaMemcpy(part->x, devicePartX, particleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->y, devicePartY, particleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->z, devicePartZ, particleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->u, devicePartU, particleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->v, devicePartV, particleSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(part->w, devicePartW, particleSize, cudaMemcpyDeviceToHost);
    double DToHTime = stopTime()*1000;

    printf("H->D: %f ms, D->H: %f ms, Kernel: %f ms\n", HToDTime, DToHTime, kernelTime);

    //@@ Free the GPU memory here
    cudaFree(devicePartX);
    cudaFree(devicePartY);
    cudaFree(devicePartZ);
    cudaFree(devicePartU);
    cudaFree(devicePartV);
    cudaFree(devicePartW);

    cudaFree(deviceXN);
    cudaFree(deviceYN);
    cudaFree(deviceZN);

    cudaFree(deviceEx);
    cudaFree(deviceEy);
    cudaFree(deviceEz);

    cudaFree(deviceBxn);
    cudaFree(deviceByn);
    cudaFree(deviceBzn);
                                                                 
    return(0); // exit succcesfully
} // end of the mover

/** Interpolation Particle --> Grid: This is for species */
void interpP2G(struct particles* part, struct interpDensSpecies* ids, struct grid* grd)
{
    
    // arrays needed for interpolation
    FPpart weight[2][2][2];
    FPpart temp[2][2][2];
    FPpart xi[2], eta[2], zeta[2];
    
    // index of the cell
    int ix, iy, iz;
    
    
    for (register long long i = 0; i < part->nop; i++) {
        
        // determine cell: can we change to int()? is it faster?
        ix = 2 + int (floor((part->x[i] - grd->xStart) * grd->invdx));
        iy = 2 + int (floor((part->y[i] - grd->yStart) * grd->invdy));
        iz = 2 + int (floor((part->z[i] - grd->zStart) * grd->invdz));
        
        // distances from node
        xi[0]   = part->x[i] - grd->XN[ix - 1][iy][iz];
        eta[0]  = part->y[i] - grd->YN[ix][iy - 1][iz];
        zeta[0] = part->z[i] - grd->ZN[ix][iy][iz - 1];
        xi[1]   = grd->XN[ix][iy][iz] - part->x[i];
        eta[1]  = grd->YN[ix][iy][iz] - part->y[i];
        zeta[1] = grd->ZN[ix][iy][iz] - part->z[i];
        
        // calculate the weights for different nodes
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    weight[ii][jj][kk] = part->q[i] * xi[ii] * eta[jj] * zeta[kk] * grd->invVOL;
        
        //////////////////////////
        // add charge density
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->rhon[ix - ii][iy - jj][iz - kk] += weight[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * weight[ii][jj][kk];
        
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add current density - Jy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        ////////////////////////////
        // add current density - Jz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->Jz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxx
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->u[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxx[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        ////////////////////////////
        // add pressure pxy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        
        /////////////////////////////
        // add pressure pxz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->u[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pxz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyy
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->v[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyy[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pyz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->v[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    ids->pyz[ix - ii][iy - jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
        
        
        /////////////////////////////
        // add pressure pzz
        for (int ii = 0; ii < 2; ii++)
            for (int jj = 0; jj < 2; jj++)
                for (int kk = 0; kk < 2; kk++)
                    temp[ii][jj][kk] = part->w[i] * part->w[i] * weight[ii][jj][kk];
        for (int ii=0; ii < 2; ii++)
            for (int jj=0; jj < 2; jj++)
                for(int kk=0; kk < 2; kk++)
                    ids->pzz[ix -ii][iy -jj][iz - kk] += temp[ii][jj][kk] * grd->invVOL;
    
    }
   
}
