#ifndef EASY_LIALG_KDEMV
#define EASY_LIALG_KDEMV

#include "./basic.h"
#include "./blas.h"
#include "./matrixdet.h"
#include <math.h>

// refer the code
// https://github.com/duncanmcn/kdepp/blob/master/include/kdepp/kdemv.h
// https://github.com/duncanmcn/kdepp/tree/master/include/kdepp

namespace EASYLINALG
{
    // data type, dim in the problem, number of samples
    template <typename T, uint Dim, uint NumSample>
    class KDEMV
    {
    public:
        KDEMV(Vec<Vec<T, NumSample>, Dim> inputData) : InputData(inputData)
        {
            init_bandwidth();
            pre_calculate_terms();
        };
        // compute probability density value
        double evaluatePDV(const Vec<T, Dim> inputSample)
        {
            double sum = 0;
            for (uint i = 0; i < NumSample; i++)
            {
                // extract each column of matrix
                // which is each sample
                Vec<T, Dim> sample;
                for (uint j = 0; j < Dim; j++)
                {
                    sample[j] = this->InputData[j][i];
                }
                //std::cout << "sample is " << std::endl;
                //sample.Show();
                //- should be operate on itsself or return a new vector?
                //if we do not copy, the inputSample will be chaned
                //auto diff = inputSample - sample;
                //instead of using the diff, just use the rvalue
                sum += kernel(inputSample - sample);
            }

            return sum / (1.0 * NumSample);
        }

    private:
        double kernel(const Vec<T, Dim> diff)
        {
            // return pow_pi_term_ * h_pow_term_ * std::exp((-1.0/2.0) * diff.transpose() * H_Inv * diff);
            //std::cout << "debug kernel diff"  << std::endl;
            //diff.Show();
            double v1 = pow_pi_term_ * h_pow_term_;
            auto DtHinv = VMMultiply(diff, this->H_Inv).dotp(diff);
            double res = v1 * std::exp((-1.0 / 2.0) * (DtHinv));
            //std::cout << "debug kernel res " << res << std::endl;
            return res;
        }

        double find_covariance(const Vec<T, NumSample> &arr1, const Vec<T, NumSample> &arr2,
                               double &mean1, double &mean2) const
        {
            double sum = 0;
            for (uint i = 0; i < NumSample; i++)
                sum = sum + (arr1[i] - mean1) * (arr2[i] - mean2);
            return (double)sum / (double)(NumSample - 1);
        }

        void init_bandwidth()
        {
            // compute cov
            // compute the mean value of input data for the convenience to get cov matrix
            for (uint i = 0; i < Dim; i++)
            {
                double tempSum = 0;
                for (uint j = 0; j < NumSample; j++)
                {
                    tempSum += this->InputData[i][j];
                }
                this->InputDataMean[i] = tempSum / (1.0 * NumSample);
            }

            Matrix<T, Dim, Dim> InitCov;
            for (uint p = 0; p < Dim; ++p)
            {
                for (uint q = p; q < Dim; ++q)
                {
                    // find the covax for the attribute p and attribute q
                    double cov = find_covariance(this->InputData[p], this->InputData[q], this->InputDataMean[p], this->InputDataMean[q]);
                    InitCov[p][q] = cov;
                    InitCov[q][p] = cov;
                }
            }
            //std::cout << "InitCov " << std::endl;
            //InitCov.Show();

            // prepare bandwidth
            // scale the H
            double n_term = std::pow(NumSample, -1.0 / (Dim + 4.0));
            //std::cout << "n_term " << n_term << std::endl;
            std::string bandwidth_method = "scott";
            if (bandwidth_method == "silverman")
            {
                double silverman_term = std::pow(4 / (Dim + 2), -1.0 / (Dim + 4.0));
                this->H = MSCALE(silverman_term * n_term * silverman_term * n_term, InitCov);
            }
            else
            {
                // scott:
                this->H = MSCALE(n_term * n_term, InitCov);
            }

            // compute inverse
            SymmInvertMatrix(H, H_Inv);

            //std::cout << "adjusted H" << std::endl;
            //H.Show();
            //std::cout << "adjusted H inv" << std::endl;
            //H_Inv.Show();

            return;
        }

        void pre_calculate_terms()
        {
            // for optimization do some pre calcs of power constants:
            pow_pi_term_ = std::pow(2 * M_PI, -1.0 * Dim / 2.0);
            // TODO, add determinant for the matrix
            double det_h = SymmDetByEigenValues(H);
            h_pow_term_ = std::pow(det_h, -1.0 / 2.0);
            //std::cout << "pow_pi_term_ " << pow_pi_term_ << " det_h " << det_h << " h_pow_term_ " << h_pow_term_ << std::endl;
        }

        // input data
        // the first component represents which dim
        // the second compoenent represents which value
        // this is different with typical matrix format which the row is sample and the column is the attribute
        Vec<Vec<T, NumSample>, Dim> InputData;
        Vec<T, Dim> InputDataMean;
        Matrix<T, Dim, Dim> H;
        Matrix<T, Dim, Dim> H_Inv;

        // Pre calculated terms for efficiency only:
        double pow_pi_term_;
        double h_pow_term_;
    };

}

#endif