/* 
 * An attempt to simulate a two dimensional ROESY spectrum with complex data points in both
 * dimensions and fft according to States-Haberkorn-Ruben. The idea is based on: Allard, P.,
 * Helgstand, M. and Hard, T., Journal of Magnetic Resonance, 129: 19-29 (1997). The code
 * aims to only use the functionalty provided by GSL as much as possible and is to a high
 * extent based on the examples provided by the GSL documentation.
 * 
 *  Compiled with: 
 *  ---------------
 *  g++ sim_roesy_gsl.cpp -lgsl -lcblas -Wall -O3
 *
 *  gnuplot script:
 *  --------------------
 *  set xrange[1000:8000]
 *  set yrange[1000:8000]
 *  set xlabel "f2 (Hz)" 
 *  set ylabel "f1 (Hz)"
 *  set hidden3d
 *  splot "spectrum.dat" u 1:2:3 with lines
 *  pause -1
 */



#include <cmath>
#include <iostream>
#include <fstream>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_fft.h>
#include <gsl/gsl_fft_complex.h>



int allard97(double t, const double x[], double dmdt[], void *params)
{
    
    /* Function implementing eq. 5 from the paper, describes cross relaxation
    between two spins during spin-lock. Returns GSL_SUCCESS, see GSL example:
    https://www.gnu.org/software/gsl/doc/html/ode-initval.html#examples. */
    

    (void)(t);                        // avoid unused parameter warning
    double *p = (double *)params;     // cast it to a pointer to double
    
    double w_a = p[0];    // offset spin A relaive to spin lock
    double r1a = p[1];
    double r2a = p[2];
    
    double w_b = p[3];    // offset spin B relaive to spin lock
    double r1b = p[4];
    double r2b = p[5];
    
    double w1 = p[6];     // spin-lock strength
    double s = p[7];      // transverse cross relaxation rate constant
    double u = p[8];      // longitsuinal cross relaxation rate constant
    
    double max = x[0];
    double may = x[1];
    double maz = x[2];
    double mbx = x[3];
    double mby = x[4];
    double mbz = x[5];
    
    const double maz_zero = 1.0;    // equilibrium magnetizations
    const double mbz_zero = 1.0;
    
    double corr_a = r1a * maz_zero + s * mbz_zero;    // correction factor, see paper
    double corr_b = s * maz_zero + r1b * mbz_zero;

    
    /* My implementation of eq. 5 from Allard, P., Helgstand, M. and Hard, T.,
     Journal of Magnetic Resonance, 129: 19-29 (1997), notice conversion to rad/s
     for some params. */
    
    dmdt[0] = -max * r2a - may * w_a * M_PI + w1 * 2 * M_PI * maz - u * mbx;         // dMAx/dt
    
    dmdt[1] = w_a * 2 * M_PI * max - r2a * may - u * mby;                            // dMAy/dt
    
    dmdt[2] = corr_a - w1 * 2 * M_PI * max - r1a * maz - s * mbz;                    // dMAz/dt
    
    dmdt[3] = -u * max - r2b * mbx - w_b * 2 * M_PI * mby + w1 * 2 * M_PI * mbz;     // dMBx/dt
    
    dmdt[4] = -u * may + w_b  * 2 * M_PI * mbx - r2b * mby;                          // dMBy/dt
    
    dmdt[5] = corr_b - s * maz - w1 * 2 * M_PI * mbx - r1b * mbz;                    // dMBz/dt

    return GSL_SUCCESS;
    
}



void do_integration(double slt, double (&mab_mag)[6], double (&parameters)[9])
{

    /* A funtion that performs the integration, returns void, 
     passing arrays by reference to it. */

    gsl_odeiv2_system sys = {allard97, NULL, 6, &parameters[0]};

    gsl_odeiv2_driver *d = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45,
                                                         1e-6, 1e-6, 0.0);
    double t = 0.0;

    for (int i = 0; i <= (slt/0.0001); i++)
    {
        double ti = i * 0.0001;
        int status = gsl_odeiv2_driver_apply(d, &t, ti, mab_mag);
    
        if (status != GSL_SUCCESS)
        {
            std::cout << "error, return " << status << std::endl;
            break;
        }
    }
    
    gsl_odeiv2_driver_free(d);
    
}



void home_made_transpose(int rows, int cols, gsl_complex_packed_array r_array)
{
    
    // Homemade transpose function to use on a gsl_complex_packed_array
    
    double *temp_array;
    temp_array = (double*) malloc(rows*cols*sizeof(double));  // new double[rows*cols];
    
    for (int i = 0; i < cols; i=i+2)
    {
        for (int j = 0; j < rows; j=j+1)
        {
            temp_array[i*rows+j*2] = r_array[i+j*cols];
            temp_array[i*rows+j*2+1] = r_array[i+j*cols+1];
        }
    }
    
    for (int i = 0; i < (rows*cols); i++)
    {
        r_array[i] = temp_array[i];
    }
    
    free(temp_array);

}



int main (int argc, char *argv[])
{
    
    /* constants and parameters */
    
    // spin A
    double freq_a = 6000.0;
    double r1a = 4.0;
    double r2a = 40.0;
    
    // spin B
    double freq_b = 4000;
    double r1b = 5.0;
    double r2b = 44.0;
    
    // spin-lock
    double O1 = 6000.0;
    double sl = 15000.0;
    double sl_time = 0.040;
    
    //cross-relaxation
    double sigma_xy = 4.0;
    double sigma_z = -1.0;
    
    // total number data points
    int td_f1 = 512;
    int td_f2 = 512;
    
    // spectral widths, increments
    double SFO1 = 600.1668163;
    double SW1 = 30.0397;
    double SWH1 = SW1 * SFO1;
    
    double SW2 = 30.0397;
    double SWH2= SW2 * SFO1;
    
    double aq_f1 = td_f1/(2*SW1*SFO1);
    double IN_F1 = aq_f1/(td_f1/2);
    
    double aq_f2 = td_f2/(2*SW2*SFO1);
    double IN_F2 = aq_f2/(td_f2/2);
    
    double lb_f1 = 30.0;
    double lb_f2 = 30.0;
    
    // reciever gain
    double RG = 6.0;
    
    
    
    // ------------------ EXPERIMENT --------------------

    double *sin_fids = NULL; 
    sin_fids = new double[(td_f1/2)*td_f2];
    double *cos_fids = NULL; 
    cos_fids = new double[(td_f1/2)*td_f2];
    
    double offset_a = freq_a-O1;
    double offset_b = freq_b-O1;

    double parameters[9] = {offset_a, r1a, r2a, offset_b,
        r1b, r2b, sl, sigma_z, sigma_xy};
    
 
    double theta_a = M_PI/2-atan(offset_a/sl);
    double theta_b = M_PI/2-atan(offset_b/sl);
    
    
    for(int i = 0; i < td_f1/2; i++)
    {
        
        // In this loop, do t1+spin-lock+t2
        
        double t1 = i*IN_F1;    // t1 (s)
        

        //------------COSINE T1 -------------
        //evolve A & B during t1 with cos term
        double cos_evo_t1_a = cos(freq_a*2*M_PI*t1)*exp(-r2a*t1);
        double cos_evo_t1_b = cos(freq_b*2*M_PI*t1)*exp(-r2b*t1);
        
        
        //------------ SINE T1 --------------
        // evolve A & B during t1 with sin term
        double sin_evo_t1_a = sin(freq_a*2*M_PI*t1)*exp(-r2a*t1);
        double sin_evo_t1_b = sin(freq_b*2*M_PI*t1)*exp(-r2b*t1);
        
        
        
        //-------- COSINE SPIN-LOCK ----------
        // cos evolved magnetization prior to spin-lock
        double max_init_sl_cos = sin_evo_t1_a;
        double may_init_sl_cos = sin(theta_a)*cos_evo_t1_a;
        double maz_init_sl_cos = abs(cos(theta_a)*cos_evo_t1_a);
        double mbx_init_sl_cos = sin_evo_t1_b;
        double mby_init_sl_cos = sin(theta_b)*cos_evo_t1_b;
        double mbz_init_sl_cos = abs(cos(theta_b)*cos_evo_t1_b);
        
        double mag_ab_cos[6] = {max_init_sl_cos, may_init_sl_cos,
                              maz_init_sl_cos, mbx_init_sl_cos,
                              mby_init_sl_cos, mbz_init_sl_cos};
        
        do_integration(sl_time, mag_ab_cos, parameters);
        
        
        //---------- SINE SPIN-LOCK ----------
        // sin evolved magnetization prior to spin-lock
        double max_init_sl_sin = cos_evo_t1_a;
        double may_init_sl_sin = sin(theta_a)*sin_evo_t1_a;
        double maz_init_sl_sin = abs(cos(theta_a)*sin_evo_t1_a);
        double mbx_init_sl_sin = cos_evo_t1_b;
        double mby_init_sl_sin = sin(theta_b)*sin_evo_t1_b;
        double mbz_init_sl_sin = abs(cos(theta_b)*sin_evo_t1_b);
        
        double mag_ab_sin[6] = {max_init_sl_sin, may_init_sl_sin,
                                maz_init_sl_sin, mbx_init_sl_sin,
                                mby_init_sl_sin, mbz_init_sl_sin};
        
        do_integration(sl_time, mag_ab_sin, parameters);
        
        
        
        
        //------------ COSINE T2 -------------
        //store fids in the packed array allocated before
        for (int j = 0; j < td_f2; j=j+2)
        {
            double t2 = j*IN_F2*0.5;
            
            double t2_cos_fid_a_real = RG*mag_ab_cos[1]*cos(freq_a*2*M_PI*t2)*exp(-r2a*2*M_PI*t2);
            double t2_cos_fid_b_real = RG*mag_ab_cos[4]*cos(freq_b*2*M_PI*t2)*exp(-r2b*2*M_PI*t2);
            
            double t2_cos_fid_a_imag = RG*mag_ab_cos[1]*sin(freq_a*2*M_PI*t2)*exp(-r2a*2*M_PI*t2);
            double t2_cos_fid_b_imag = RG*mag_ab_cos[4]*sin(freq_b*2*M_PI*t2)*exp(-r2b*2*M_PI*t2);
            
            // OBS!!! incrementing j by 2 to store real and imag part
            // as pairs in packed array, the same goes for sine t2
            
            cos_fids[i*td_f2+j] = t2_cos_fid_a_real+t2_cos_fid_b_real;
            cos_fids[i*td_f2+j+1] = t2_cos_fid_a_imag+t2_cos_fid_b_imag;
        }
        
        
        //------------- SINE T2 -------------
        // store fids in packed array allocated before
        for (int j = 0; j < td_f2; j=j+2)
        {
            double t2 = j*IN_F2*0.5;
            
            double t2_sin_fid_a_real = RG*mag_ab_sin[1]*cos(freq_a*2*M_PI*t2)*exp(-r2a*2*M_PI*t2);
            double t2_sin_fid_b_real = RG*mag_ab_sin[4]*cos(freq_b*2*M_PI*t2)*exp(-r2b*2*M_PI*t2);
            
            double t2_sin_fid_a_imag = RG*mag_ab_sin[1]*sin(freq_a*2*M_PI*t2)*exp(-r2a*2*M_PI*t2);
            double t2_sin_fid_b_imag = RG*mag_ab_sin[4]*sin(freq_b*2*M_PI*t2)*exp(-r2b*2*M_PI*t2);
            
            sin_fids[i*td_f2+j] = t2_sin_fid_a_real+t2_sin_fid_b_real;
            sin_fids[i*td_f2+j+1] = t2_sin_fid_a_imag+t2_sin_fid_b_imag;
        }
        
    }
    

    // Apply window function in f2 dimension

    for (int i = 0;  i < td_f1/2; i++)
    {
        for (int j = 0; j < td_f2; j=j+2)
        {
         double t2 = j*IN_F2*0.5;
         
         cos_fids[i*td_f2+j] = cos_fids[i*td_f2+j] * exp(-lb_f2*t2);     // real part
         cos_fids[i*td_f2+j+1] = cos_fids[i*td_f2+j+1] * exp(-lb_f2*t2); // imag part
         
         sin_fids[i*td_f2+j] = sin_fids[i*td_f2+j] * exp(-lb_f2*t2);     // real part
         sin_fids[i*td_f2+j+1] = sin_fids[i*td_f2+j+1] * exp(-lb_f2*t2); // real part
         
        }
    }
    
    // ------ FFT PROCESSING, STATES-HABERKORN-RUBEN -----
    
    
    for(int i = 0; i < (td_f1/2); i++)
    {
        // t2 fids are stored continous, as a (td_f1/2) * td_f2 row major matrix.
        // even numbered positions in array will contain real part and odd the imaginary
        // part for the data points. The real part of cos_fids and sin_fids will later
        // form the complex data points for the t1 dimension together.
        
        // do fft for each row in matrix
        gsl_fft_complex_radix2_transform(&cos_fids[i*td_f2], 1, td_f2/2, gsl_fft_forward);
        gsl_fft_complex_radix2_transform(&sin_fids[i*td_f2], 1, td_f2/2, gsl_fft_forward);
    }
    
    
    // new complex matrix
    double *comp_matrix = NULL;
    comp_matrix = new double[(td_f1/2)*td_f2];
    
    for(int i = 0; i < td_f2*(td_f1/2); i = i+2)
    {
        // transfer real parts of tranformed cos_fids
        // and sin_fids to thenew complex array
        comp_matrix[i] = cos_fids[i];
        comp_matrix[i+1] = sin_fids[i];
    }
    
    
    // transpose
    home_made_transpose(td_f1/2, td_f2, comp_matrix);

    
    // Apply window function in f1 dimension

    for (int i = 0;  i < td_f2/2; i++)
    {
        for (int j = 0; j < td_f1; j=j+2)
        {
         double t1 = j*IN_F1*0.5;
         
         comp_matrix[i*td_f1+j] = comp_matrix[i*td_f1+j]*exp(-lb_f1*t1);     // real part
         comp_matrix[i*td_f1+j+1] = comp_matrix[i*td_f1+j+1]*exp(-lb_f1*t1); // imag part
         
        }
    }

    
    for(int i = 0; i < (td_f1/2); i++)
    {
        // do horizontal fft along again, along f1 since it transposed
        gsl_fft_complex_radix2_transform(&comp_matrix[i*td_f1], 1, td_f1/2, gsl_fft_forward);
    }

      
    // transpose again to have direct dimension along horizontal axis in spectrum
    home_made_transpose(td_f1/2, td_f2, comp_matrix);
    
     
    // calcute frequencies for the axis
    double start_freq_f1 = 0.0;
    double start_freq_f2 = 0.0;
    
    double inc =  ( 1/(2*IN_F2) ) / (td_f2/4);

    std::ofstream outpf;
    outpf.open("spectrum.dat");
    
    for (int i = 0; i < td_f1/4; i++)
    {
        for (int j = 0; j < td_f2/2; j=j+2)
        {
            // 3 columns = f2 frequncy, f1 frequncy, intensity. Suitable for gnuplot 
            outpf << start_freq_f2 <<" "<< start_freq_f1 <<" "<< comp_matrix[j+i*td_f2] << std::endl;
            start_freq_f2 = start_freq_f2 + inc;
        }
        start_freq_f2 = 0;
        start_freq_f1 = start_freq_f1 + inc;
        outpf << std::endl;
    }
    
    outpf.close();
        
    delete comp_matrix;
    delete sin_fids;
    delete cos_fids;

    return 0;
}



