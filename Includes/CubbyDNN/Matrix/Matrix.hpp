#ifndef CUBBYDNN_MATRIX_HPP
#define CUBBYDNN_MATRIX_HPP

#include <functional>
#include <cassert>

namespace CubbyDNN {
    template<typename T>
    class Matrix {
    private:
        T *mat_ptr = nullptr;
        size_t _row, _col;
    public:
        explicit Matrix(size_t row, size_t col=1, bool is_init=true, T init_val=0, typename std::enable_if_t<std::is_floating_point<T>::value || std::is_integral<T>::value>* = nullptr) {
            _row = row;
            _col = col;
            mat_ptr = new T[_row * _col];
            if(is_init)
                std::fill(mat_ptr, mat_ptr+sizeof(T)*_row*_col, init_val);
        }

        ~Matrix() {
            if(mat_ptr != nullptr)
                delete mat_ptr;
        }

        inline size_t row(){
            return _row;
        }

        inline size_t col(){
            return _col;
        }

        inline T& operator ()(size_t i, size_t j){
            assert(i<_row && j<_col);
            return mat_ptr[i*_col + j];
        }

        template<typename RT>
        Matrix operator *(const Matrix<RT>& r) const{
            assert(_col == r._row);
            Matrix ret(_row, r._col);
            for(size_t i=0;i<_row;++i){
                for(size_t j=0;j<r._col;++j){
                    for(size_t w=0;w<_col;++w){
                        ret.mat_ptr[i*r._col+j] += mat_ptr[i*_col + w] * r.mat_ptr[w*r._col+j];
                    }
                }
            }
            return ret;
        }

        template<typename RT>
        Matrix operator %(const Matrix<RT>& r) const {
            assert(_row == r._row and _col == r._col);
            Matrix ret(_row, _col, false);
            for(size_t i=0;i<_row*_col;++i){
                ret.mat_ptr[i] = mat_ptr[i] * r.mat_ptr[i];
            }
            return ret;
        }

        template<typename RT>
        Matrix operator +(const Matrix<RT>& r) const{
            assert(_row == r._row and _col == r._col);
            Matrix ret(_row, _col, false);
            for(size_t i=0;i<_row*_col;++i){
                ret.mat_ptr[i] = mat_ptr[i] + r.mat_ptr[i];
            }
            return ret;
        }

        template<typename RT>
        Matrix operator -(const Matrix<RT>& r) const{
            assert(_row == r._row and _col == r._col);
            Matrix ret(_row, _col, false);
            for(size_t i=0;i<_row*_col;++i){
                ret.mat_ptr[i] = mat_ptr[i] - r.mat_ptr[i];
            }
            return ret;
        }

        template<typename RT>
        Matrix operator /(const Matrix<RT>& r) const{
            assert(_row == r._row and _col == r._col);
            Matrix ret(_row, _col, false);
            for(size_t i=0;i<_row*_col;++i){
                ret.mat_ptr[i] = mat_ptr[i] / r.mat_ptr[i];
            }
            return ret;
        }

        void for_each(const std::function<T(T)>& func) const{
            for(size_t i=0;i<_row*_col;++i){
                mat_ptr[i] = func(mat_ptr[i]);
            }
        }

        void for_each(const std::function<T(size_t row, size_t col, T)>& func){
            for(size_t i=0;i<_row;++i){
                for(size_t j=0;j<_col;++j){
                    mat_ptr[i*_col + j] = func(i, j, mat_ptr[i*_col + j]);
                }
            }
        }

        void trans(){
            if(_row == _col){
                //inplace
                for(size_t i=1;i<_row;++i){
                    for(size_t j=0;j<i;++j){
                        std::swap(mat_ptr[i*_col+j], mat_ptr[j*_col+i]);
                    }
                }
                return;
            }
            T* new_ptr = new T[_row * _col];
            for(size_t i=0;i<_row;++i){
                for(size_t j=0;j<_col;++j){
                    new_ptr[j*_row + i] = mat_ptr[i*_col + j];
                }
            }
            delete mat_ptr;
            mat_ptr = new_ptr;
        }

        template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>*>
        friend Matrix<MT> operator +(const Matrix<MT>& l, NT r);

        template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>*>
        friend Matrix<MT> operator +(NT l, const Matrix<MT>& r);

        template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>*>
        friend Matrix<MT> operator *(const Matrix<MT>& l, NT r);

        template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>*>
        friend Matrix<MT> operator *(NT l, const Matrix<MT>& r);

        template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>*>
        friend Matrix<MT> operator -(const Matrix<MT>& l, NT r);

        template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>*>
        friend Matrix<MT> operator -(NT l, const Matrix<MT>& r);

        template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>*>
        friend Matrix<MT> operator /(const Matrix<MT>& l, NT r);

        template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>*>
        friend Matrix<MT> operator /(NT l, const Matrix<MT>& r);
    };

    //operators between scalar

    template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>* = nullptr>
    Matrix<MT> operator +(const Matrix<MT>& l, const NT r){
        Matrix<MT> ret(l._row, l._col, false);
        for(size_t i=0;i<ret._row*ret._col;++i){
            ret.mat_ptr[i] = l.mat_ptr[i] + r;
        }
        return ret;
    }

    template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>* = nullptr>
    Matrix<MT> operator +(const NT l, const Matrix<MT>& r){
        Matrix<MT> ret(r._row, r._col, false);
        for(size_t i=0;i<ret._row*ret._col;++i){
            ret.mat_ptr[i] = r.mat_ptr[i] + l;
        }
        return ret;
    }

    template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>* = nullptr>
    Matrix<MT> operator *(const Matrix<MT>& l, const NT r){
        Matrix<MT> ret(l._row, l._col, false);
        for(size_t i=0;i<ret._row*ret._col;++i){
            ret.mat_ptr[i] = l.mat_ptr[i] * r;
        }
        return ret;
    }

    template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>* = nullptr>
    Matrix<MT> operator *(const NT l, const Matrix<MT>& r){
        Matrix<MT> ret(r._row, r._col, false);
        for(size_t i=0;i<ret._row*ret._col;++i){
            ret.mat_ptr[i] = r.mat_ptr[i] * l;
        }
        return ret;
    }

    template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>* = nullptr>
    Matrix<MT> operator -(const Matrix<MT>& l, const NT r){
        Matrix<MT> ret(l._row, l._col, false);
        for(size_t i=0;i<ret._row*ret._col;++i){
            ret.mat_ptr[i] = l.mat_ptr[i] - r;
        }
        return ret;
    }

    template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>* = nullptr>
    Matrix<MT> operator -(const NT l, const Matrix<MT>& r){
        Matrix<MT> ret(r._row, r._col, false);
        for(size_t i=0;i<ret._row*ret._col;++i){
            ret.mat_ptr[i] = l - r.mat_ptr[i];
        }
        return ret;
    }

    template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>* = nullptr>
    Matrix<MT> operator /(const Matrix<MT>& l, const NT r){
        Matrix<MT> ret(l._row, l._col, false);
        for(size_t i=0;i<ret._row*ret._col;++i){
            ret.mat_ptr[i] = l.mat_ptr[i] / r;
        }
        return ret;
    }

    template<typename MT, typename NT, std::enable_if_t<std::is_floating_point<NT>::value || std::is_integral<NT>::value>* = nullptr>
    Matrix<MT> operator /(const NT l, const Matrix<MT>& r){
        Matrix<MT> ret(r._row, r._col, false);
        for(size_t i=0;i<ret._row*ret._col;++i){
            ret.mat_ptr[i] = l / r.mat_ptr[i];
        }
        return ret;
    }
}


#endif //CUBBYDNN_MATRIX_HPP
