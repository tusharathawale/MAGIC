

#ifndef EASY_LIALG_KDE
#define EASY_LIALG_KDE

#include "./basic.h"
#include "./blas.h"

//reference 
//https://billc.io/2023/01/kde-from-scratch/
//https://aakinshin.net/posts/kde-bw/
//https://medium.com/@BillChen2k/kernel-density-estimation-with-python-from-scratch-c200b187b6c4
//https://github.com/billchen2k/KDEBandwidth (it seems that Silverman is more accurate)

namespace EASYLINALG
{
    template <typename T, uint Size>
    LIAG_FUNC_MACRO float BandWidthScott(const Vec<T, Size> &input)
    {
        bool sampleStdev = true;
        double stdev = input.GetStdev(sampleStdev);
        int len = input.NUM_COMPONENTS;
        return 3.49 * stdev * pow(len, (-0.333));
    }

    LIAG_FUNC_MACRO float GaussianKernel(float input)
    {
        return (1 / (sqrt(2 * M_PI))) * exp(-0.5 * input * input);
    }

    // input: data vector,dedicated point, bandwidth(default parameter)
    // output kde results at the dedicated point
    template <typename T, uint Size>
    LIAG_FUNC_MACRO double KDE1D(const Vec<T, Size> &inputSample, float dedicatedPoint)
    {
        // choose the bandWidth according to input if it uses default value
        float h = BandWidthScott(inputSample);

        double kde = 0;
        // go through each input value, compuet kde value
        for (int i = 0; i < inputSample.NUM_COMPONENTS; i++)
        {
            double kernelInput = (dedicatedPoint - inputSample[i]) / h;
            kde = kde + GaussianKernel(kernelInput);
        }

        kde = kde / (inputSample.NUM_COMPONENTS * 1.0 * h);

        return kde;
    }

    // CDF of the KDE, Phi is the error function
    // pdf
    // f(x) = 1/nh sum K( (x-x_i)/h )
    // cdf t= (x-x_i)/h
    // F(t<X) = F( (x-x_i)/h<X )

    // the input value X means
    // that we want to caculate the function where F(x<X)
    template <typename T, uint Size>
    LIAG_FUNC_MACRO double KDECDF1D(const Vec<T, Size> &inputSample, float X)
    {
        float h = BandWidthScott(inputSample);
        //float h = 1.0;
        //std::cout << "h is " << h << std::endl;
        double Phi = 0;
        for (int i = 0; i < inputSample.NUM_COMPONENTS; i++)
        {
            double t = (X-inputSample[i])/h;
            //std::cout << "t " << t << std::endl;
            // for cuda math function, there is erf
            double phi = 0.5 * (1 + std::erf(t / std::sqrt(2)));
            //std::cout << "phi " << phi << std::endl;
            Phi = Phi + phi;
        }
        Phi = Phi / (inputSample.NUM_COMPONENTS );
        return Phi;
    }
}

#endif