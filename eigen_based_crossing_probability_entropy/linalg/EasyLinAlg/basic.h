#ifndef EASY_LIALG_BASIC
#define EASY_LIALG_BASIC

#include <iostream>
#include <assert.h>
#include <cmath>
#include <type_traits>

namespace EASYLINALG
{

// basic defination of template based matrix and vector

// set as necessary header as needed
// such as one using on GPU
#define LIAG_FUNC_MACRO __attribute__((visibility("default")))

    template <typename T, uint Size>
    class Vec
    {
    public:
        static const int NUM_COMPONENTS = Size;
        T Components[NUM_COMPONENTS];

        LIAG_FUNC_MACRO Vec() {}

        LIAG_FUNC_MACRO Vec(T init)
        {
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                this->Components[i] = init;
            }
        }

        LIAG_FUNC_MACRO Vec(const Vec &src)
        {
            assert(this->NUM_COMPONENTS == src.NUM_COMPONENTS);
            for (uint i = 0; i < Size; ++i)
            {
                this->Components[i] = src[i];
            }
        }

        // TODO
        // the += operator should use the original vector
        // the single + or - should just allocate a new vector
        LIAG_FUNC_MACRO Vec<T, Size> operator+(const Vec &t) const
        {
            assert(this->NUM_COMPONENTS == t.NUM_COMPONENTS);
            Vec<T, Size> result;
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                result[i] = this->Components[i] + t.Components[i];
            }
            return result;
        }

        LIAG_FUNC_MACRO Vec<T, Size> operator-(const Vec &t) const
        {
            assert(this->NUM_COMPONENTS == t.NUM_COMPONENTS);
            Vec<T, Size> result;
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                result[i] = this->Components[i] - t.Components[i];
            }
            return result;
        }

        // getting element by [] operator
        // return a reference, so the value can be updated further
        LIAG_FUNC_MACRO T &operator[](int index)
        {
            assert(index >= 0);
            assert(index < this->NUM_COMPONENTS);
            return this->Components[index];
        }

        // if not adding this, there are some error such as
        /*
        /Users/zw1/Documents/cworkspace/src/EasyLinalg/StaticMemTemplate/mathTemplate.h:141:42: error: no viable overloaded operator[] for type 'const Vec<Vec<double, NUM_COLUMNS>, NUM_ROWS>'
                    this->Components[i][j] = t.Components[i][j];
                                             ^~~~~~~~~~~~ ~
        /Users/zw1/Documents/cworkspace/src/EasyLinalg/StaticMemTemplate/mathTemplate.h:343:15: note: in instantiation of member function 'Matrix<double, 3, 3>::operator=' requested here
                Q = MatrixMultiply(Q, MList[k]);
                  ^
        /Users/zw1/Documents/cworkspace/src/EasyLinalg/StaticMemTemplate/test.cpp:163:5: note: in instantiation of function template specialization 'Householder<double, 3U, 3U>' requested here
        Householder(x,Q,R);
        ^
        /Users/zw1/Documents/cworkspace/src/EasyLinalg/StaticMemTemplate/test.cpp:187:5: note: in instantiation of function template specialization 'testQRInner<3U>' requested here
        testQRInner<3>();
        ^
        /Users/zw1/Documents/cworkspace/src/EasyLinalg/StaticMemTemplate/mathTemplate.h:18:8: note: candidate function not viable: 'this' argument has type 'const Vec<Vec<double, NUM_COLUMNS>, NUM_ROWS>', but method is not marked const
        T &operator[](const int index)

        The failure happens because Non-const functions can only be called by non-const objects.
        However, when a function is declared as const, it can be called on any type of object.

        still curious about results here, some cases, the object that call the [] is a const object
        */
        LIAG_FUNC_MACRO const T &operator[](int index) const
        {
            assert(index >= 0);
            assert(index < this->NUM_COMPONENTS);
            return this->Components[index];
        }

        LIAG_FUNC_MACRO Vec<T, Size> &operator=(const Vec<T, Size> &t)
        {
            assert(this->NUM_COMPONENTS == t.NUM_COMPONENTS);
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                this->Components[i] = t.Components[i];
            }
            return *this;
        }

        LIAG_FUNC_MACRO Vec<T, Size> &operator=(Vec<T, Size> &t)
        {
            assert(this->NUM_COMPONENTS == t.NUM_COMPONENTS);
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                this->Components[i] = t.Components[i];
            }
            return *this;
        }

        // this is only used for single vector
        // the vector can be in inserted form
        // so we do not set the init value of vector as zero
        LIAG_FUNC_MACRO void InitZero()
        {
            for (int i = 0; i < this->NUM_COMPONENTS; i++)
            {
                this->Components[i] = 0;
            }
        }

        LIAG_FUNC_MACRO void Show()
        {
            for (int i = 0; i < this->NUM_COMPONENTS; i++)
            {
                printf("%lf ", 1.0 * this->Components[i]);
            }
            printf("\n");
        }

        // ||x||
        LIAG_FUNC_MACRO double GetNorm()
        {
            double sum = 0;
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
                sum = sum + this->Components[i] * this->Components[i];
            return sqrt(sum);
        }

        LIAG_FUNC_MACRO double GetStdev(bool sampleStdev = false) const
        {
            float sum = 0.0, mean, standardDeviation = 0.0;
            int i;

            for (i = 0; i < this->NUM_COMPONENTS; ++i)
            {
                sum += this->Components[i];
            }

            mean = sum / this->NUM_COMPONENTS;

            for (i = 0; i < this->NUM_COMPONENTS; ++i)
            {
                standardDeviation += pow(this->Components[i] - mean, 2);
            }

            if (sampleStdev)
            {
                return sqrt(standardDeviation / (this->NUM_COMPONENTS - 1));
            }

            return sqrt(standardDeviation / this->NUM_COMPONENTS);
        }

        // y = y/d
        LIAG_FUNC_MACRO void Div(double d)
        {
            // if the d is a value close to zero
            // just set results to 0
            assert((d - 0.0) > 0.00001);
            for (uint i = 0; i < Size; i++)
            {
                this->Components[i] = static_cast<T>(1.0 * this->Components[i] / d);
            }
            return;
        }

        LIAG_FUNC_MACRO bool Equal(const Vec &t)
        {
            if (this->NUM_COMPONENTS != t.NUM_COMPONENTS)
            {
                return false;
            }

            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                if (fabs(this->Components[i] - t.Components[i]) > 0.0001)
                {
                    return false;
                }
            }
            return true;
        }

        LIAG_FUNC_MACRO void Normalize()
        {
            double sum = 0;
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                sum += this->Components[i] * this->Components[i];
            }
            // std::cout << "sum " << sum << std::endl;
            // this->Show();
            assert(fabs(sum - 0.0) > 0.0);
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                this->Components[i] = this->Components[i] / sqrt(sum);
            }
        }

        LIAG_FUNC_MACRO void Sort()
        {
            // sort the current array
            // from large element to small element
            T temp;
            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                for (uint j = 0; j < this->NUM_COMPONENTS - 1 - i; j++)
                {
                    // from largest one to the smallest one
                    if (this->Components[j] < this->Components[j + 1])
                    {
                        temp = this->Components[j];
                        this->Components[j] = this->Components[j + 1];
                        this->Components[j + 1] = temp;
                    }
                }
            }
        }

        // dot product of another vector
        // assuming vector is
        LIAG_FUNC_MACRO T dotp(const Vec &t)
        {
            T productV = 0;
            assert(this->NUM_COMPONENTS == t.NUM_COMPONENTS);

            for (uint i = 0; i < this->NUM_COMPONENTS; i++)
            {
                productV += this->Components[i] * t.Components[i];
            }
            return productV;
        }
    };

    template <typename T, uint RowSize, uint ColSize>
    class Matrix
    {
    public:
        static constexpr int NUM_ROWS = RowSize;
        static constexpr int NUM_COLUMNS = ColSize;
        Vec<Vec<T, NUM_COLUMNS>, NUM_ROWS> Components;

        LIAG_FUNC_MACRO Matrix()
        {
            for (uint i = 0; i < this->NUM_ROWS; i++)
            {
                for (uint j = 0; j < this->NUM_COLUMNS; j++)
                {
                    this->Components[i][j] = 0;
                }
            }
        };

        LIAG_FUNC_MACRO bool IsUpperTriangular(double tol) const
        {
            // For now, only treat square matricies.
            if (this->NUM_COLUMNS != this->NUM_ROWS)
            {
                return false;
            }

            for (uint i = 0; i < this->NUM_COLUMNS; i++)
            {
                for (uint j = 0; j < i; j++)
                {
                    if (fabs(this->Components[i][j]) > tol)
                    {
                        // std::cout << fabs(this->Components[i][j]) << " tol " << tol << " " << bool(fabs(this->Components[i][j])>tol) << std::endl;
                        return false;
                    }
                }
            }
            return true;
        }

        LIAG_FUNC_MACRO bool IsEqual(const Matrix &t, double tol = 0.00001)
        {
            if (this->NUM_ROWS != t.NUM_ROWS || this->NUM_COLUMNS != t.NUM_COLUMNS)
            {
                return false;
            }
            for (uint i = 0; i < this->NUM_ROWS; i++)
            {
                for (uint j = 0; j < this->NUM_COLUMNS; j++)
                {
                    if (fabs(this->Components[i][j] - t.Components[i][j]) > tol)
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        LIAG_FUNC_MACRO Vec<T, NUM_COLUMNS> &operator[](uint rowIndex)
        {
            assert(rowIndex >= 0);
            assert(rowIndex < NUM_ROWS);
            return this->Components[rowIndex];
        }

        LIAG_FUNC_MACRO const Vec<T, NUM_COLUMNS> &operator[](uint rowIndex) const
        {
            assert(rowIndex >= 0);
            assert(rowIndex < NUM_ROWS);
            return this->Components[rowIndex];
        }

        LIAG_FUNC_MACRO Matrix &operator-(Matrix &t)
        {
            assert(this->NUM_ROWS == t.NUM_ROWS);
            assert(this->NUM_COLUMNS == t.NUM_COLUMNS);

            for (uint i = 0; i < this->NUM_ROWS; i++)
            {
                for (uint j = 0; j < this->NUM_COLUMNS; j++)
                {
                    this->Components[i][j] = this->Components[i][j] - t.Components[i][j];
                }
            }
            return *this;
        }

        LIAG_FUNC_MACRO Matrix (const Matrix &src)
        {
            assert(this->NUM_ROWS == src.NUM_ROWS);
            assert(this->NUM_COLUMNS == src.NUM_COLUMNS);

            for (uint i = 0; i < this->NUM_ROWS; i++)
            {
                for (uint j = 0; j < this->NUM_COLUMNS; j++)
                {
                    this->Components[i][j] = src.Components[i][j];
                }
            }
        }
        
        LIAG_FUNC_MACRO Matrix &operator+(Matrix &t)
        {
            assert(this->NUM_ROWS == t.NUM_ROWS);
            assert(this->NUM_COLUMNS == t.NUM_COLUMNS);

            for (uint i = 0; i < this->NUM_ROWS; i++)
            {
                for (uint j = 0; j < this->NUM_COLUMNS; j++)
                {
                    this->Components[i][j] = this->Components[i][j] + t.Components[i][j];
                }
            }
            return *this;
        }

        LIAG_FUNC_MACRO Matrix<T, RowSize, ColSize> &operator=(Matrix<T, RowSize, ColSize> &t)
        {
            assert(this->NUM_ROWS == t.NUM_ROWS);
            assert(this->NUM_COLUMNS == t.NUM_COLUMNS);

            for (uint i = 0; i < this->NUM_ROWS; i++)
            {
                for (uint j = 0; j < this->NUM_COLUMNS; j++)
                {
                    this->Components[i][j] = t.Components[i][j];
                }
            }
            return *this;
        }

        LIAG_FUNC_MACRO Matrix<T, RowSize, ColSize> &operator=(const Matrix<T, RowSize, ColSize> &t)
        {
            assert(this->NUM_ROWS == t.NUM_ROWS);
            assert(this->NUM_COLUMNS == t.NUM_COLUMNS);

            for (uint i = 0; i < this->NUM_ROWS; i++)
            {
                for (uint j = 0; j < this->NUM_COLUMNS; j++)
                {
                    this->Components[i][j] = t.Components[i][j];
                }
            }
            return *this;
        }

        LIAG_FUNC_MACRO void InitEye()
        {
            assert(this->NUM_ROWS == this->NUM_COLUMNS);
            for (int i = 0; i < this->NUM_ROWS; i++)
            {
                for (int j = 0; j < this->NUM_COLUMNS; j++)
                {
                    if (i == j)
                    {
                        this->Components[i][j] = 1;
                    }
                    else
                    {
                        this->Components[i][j] = 0;
                    }
                }
            }
        }

        LIAG_FUNC_MACRO void Show()
        {
            for (int i = 0; i < this->NUM_ROWS; i++)
            {
                for (int j = 0; j < this->NUM_COLUMNS; j++)
                {
                    printf(" %8.7f", 1.0 * this->Components[i][j]);
                }
                printf("\n");
            }
            printf("\n");
        }

        LIAG_FUNC_MACRO void Transpose()
        {
            for (int i = 0; i < this->NUM_ROWS; i++)
            {
                for (int j = 0; j < i; j++)
                {
                    T t = this->Components[i][j];
                    this->Components[i][j] = this->Components[j][i];
                    this->Components[j][i] = t;
                }
            }
        }

        // only works for the symmetric matrix
        LIAG_FUNC_MACRO Matrix GetMinor(uint d)
        {
            assert(this->NUM_ROWS == this->NUM_COLUMNS);
            Matrix minor;
            for (uint i = 0; i < d; i++)
            {
                minor.Components[i][i] = 1;
            }

            // start form the dth position
            // make it same with the original data

            for (uint i = d; i < NUM_ROWS; i++)
            {
                for (uint j = d; j < NUM_COLUMNS; j++)
                {
                    minor.Components[i][j] = this->Components[i][j];
                }
            }

            return minor;
        }

        // take c-th column of matrix, put in vector
        LIAG_FUNC_MACRO Vec<T, NUM_COLUMNS> GetColum(uint c)
        {
            Vec<T, NUM_COLUMNS> vec;
            for (uint i = 0; i < this->NUM_ROWS; i++)
                vec[i] = this->Components[i][c];
            return vec;
        }
    };

}

#endif
