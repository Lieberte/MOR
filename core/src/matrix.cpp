#include "matrix.hpp"
#include "parallelContext.hpp"
#include <cmath>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <algorithm>

namespace shonClooud::shonTA::rom::framework {
namespace core {

matrix::matrix(int globalRows,
               int globalCols,
               const parallelContext& ctx,
               bool distributed)
    : dCtx(std::make_shared<const parallelContext>(ctx))
    , dLocalRows(0)
    , dLocalCols(globalCols)
    , dGlobalRows(globalRows)
    , dGlobalCols(globalCols)
    , dDistributed(distributed)
{
    if (dDistributed) {
        dLocalRows = dCtx->splitDimension(globalRows);
    } else {
        dLocalRows = globalRows;
    }
    
    double* rawData = new double[dLocalRows * dLocalCols];
    std::fill_n(rawData, dLocalRows * dLocalCols, 0.0);
    dData = std::shared_ptr<double>(rawData, [](double* p) { delete[] p; });
}

matrix::matrix(double* data,
               int localRows,
               int localCols,
               int globalRows,
               int globalCols,
               const parallelContext& ctx,
               bool distributed,
               bool copyData)
    : dCtx(std::make_shared<const parallelContext>(ctx))
    , dLocalRows(localRows)
    , dLocalCols(localCols)
    , dGlobalRows(globalRows)
    , dGlobalCols(globalCols)
    , dDistributed(distributed)
{
    int dataSize = localRows * localCols;
    if (copyData) {
        double* rawData = new double[dataSize];
        std::memcpy(rawData, data, dataSize * sizeof(double));
        dData = std::shared_ptr<double>(rawData, [](double* p) { delete[] p; });
    } else {
        dData = std::shared_ptr<double>(data, [](double*) {});
    }
}

matrix::matrix(const matrix& other)
    : dCtx(other.dCtx)
    , dLocalRows(other.dLocalRows)
    , dLocalCols(other.dLocalCols)
    , dGlobalRows(other.dGlobalRows)
    , dGlobalCols(other.dGlobalCols)
    , dDistributed(other.dDistributed)
{
    int dataSize = dLocalRows * dLocalCols;
    double* rawData = new double[dataSize];
    std::memcpy(rawData, other.dData.get(), dataSize * sizeof(double));
    dData = std::shared_ptr<double>(rawData, [](double* p) { delete[] p; });
}

matrix::matrix(matrix&& other) noexcept
    : dCtx(std::move(other.dCtx))
    , dData(std::move(other.dData))
    , dLocalRows(other.dLocalRows)
    , dLocalCols(other.dLocalCols)
    , dGlobalRows(other.dGlobalRows)
    , dGlobalCols(other.dGlobalCols)
    , dDistributed(other.dDistributed)
{
    other.dLocalRows = 0;
    other.dLocalCols = 0;
    other.dGlobalRows = 0;
    other.dGlobalCols = 0;
    other.dDistributed = false;
}

matrix::~matrix() = default;

matrix& matrix::operator=(const matrix& rhs)
{
    if (this != &rhs) {
        dCtx = rhs.dCtx;
        dLocalRows = rhs.dLocalRows;
        dLocalCols = rhs.dLocalCols;
        dGlobalRows = rhs.dGlobalRows;
        dGlobalCols = rhs.dGlobalCols;
        dDistributed = rhs.dDistributed;
        
        int dataSize = dLocalRows * dLocalCols;
        double* rawData = new double[dataSize];
        std::memcpy(rawData, rhs.dData.get(), dataSize * sizeof(double));
        dData = std::shared_ptr<double>(rawData, [](double* p) { delete[] p; });
    }
    return *this;
}

matrix& matrix::operator=(matrix&& rhs) noexcept
{
    if (this != &rhs) {
        dCtx = std::move(rhs.dCtx);
        dData = std::move(rhs.dData);
        dLocalRows = rhs.dLocalRows;
        dLocalCols = rhs.dLocalCols;
        dGlobalRows = rhs.dGlobalRows;
        dGlobalCols = rhs.dGlobalCols;
        dDistributed = rhs.dDistributed;
        
        rhs.dLocalRows = 0;
        rhs.dLocalCols = 0;
        rhs.dGlobalRows = 0;
        rhs.dGlobalCols = 0;
        rhs.dDistributed = false;
    }
    return *this;
}

int matrix::numGlobalRows() const
{
    if (dDistributed && dGlobalRows == 0) {
        std::vector<int> offsets;
        dCtx->getGlobalOffsets(dLocalRows, offsets);
        if (!offsets.empty()) {
            int localGlobalRows = offsets.back() + dLocalRows;
            return localGlobalRows;
        }
    }
    return dGlobalRows;
}

double matrix::norm() const
{
    return frobeniusNorm();
}

double matrix::frobeniusNorm() const
{
    double localSum = 0.0;
    int dataSize = dLocalRows * dLocalCols;
    for (int i = 0; i < dataSize; ++i) {
        double val = dData.get()[i];
        localSum += val * val;
    }
    
    if (dDistributed) {
        dCtx->allReduce(&localSum, 1, MPI_SUM);
    }
    
    return std::sqrt(localSum);
}

void matrix::scale(double alpha)
{
    int dataSize = dLocalRows * dLocalCols;
    for (int i = 0; i < dataSize; ++i) {
        dData.get()[i] *= alpha;
    }
}

void matrix::setZero()
{
    std::fill_n(dData.get(), dLocalRows * dLocalCols, 0.0);
}

void matrix::matVecMult(const vector& x, vector& y) const
{
    if (dLocalCols != x.numLocalRows() || dLocalRows != y.numLocalRows()) { 
        return;
    }
    
    y.setZero();
    
    for (int i = 0; i < dLocalRows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dLocalCols; ++j) {
            sum += item(i, j) * x[j];
        }
        y[i] = sum;
    }
    
    if (dDistributed) {
    }
}

void matrix::matVecMultAdd(const vector& x, vector& y, double alpha) const
{
    if (dLocalCols != x.numLocalRows() || dLocalRows != y.numLocalRows()) {
        return;
    }
    
    for (int i = 0; i < dLocalRows; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dLocalCols; ++j) {
            sum += item(i, j) * x[j];
        }
        y[i] = alpha * sum + y[i];
    }
    
    if (dDistributed) {
    }
}

void matrix::mult(const matrix& other, matrix& result) const
{
    if (other.dDistributed) {
        return;
    }
    
    if (dLocalCols != other.dLocalRows || dLocalRows != result.dLocalRows || 
        other.dGlobalCols != result.dGlobalCols) {
        return;
    }
    
    result.setZero();
    
    for (int i = 0; i < dLocalRows; ++i) {
        for (int j = 0; j < other.dLocalCols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < dLocalCols; ++k) {
                sum += item(i, k) * other.item(k, j);
            }
            result.item(i, j) = sum;
        }
    }
}

void matrix::transposeMult(const matrix& other, matrix& result) const
{
    if (dLocalRows != other.dLocalRows || dLocalCols != result.dLocalRows ||
        other.dGlobalCols != result.dGlobalCols) {
        return;
    }
    
    result.setZero();
    
    for (int i = 0; i < dLocalCols; ++i) {
        for (int j = 0; j < other.dLocalCols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < dLocalRows; ++k) {
                sum += item(k, i) * other.item(k, j);
            }
            result.item(i, j) = sum;
        }
    }
    
    if (dDistributed && dCtx->getSize() > 1) {
        int resultSize = dLocalCols * other.dLocalCols;
        dCtx->allReduce(result.dData.get(), resultSize, MPI_SUM);
    }
}

void matrix::transposeMult(const vector& other, vector& result) const
{
    if (dLocalRows != other.numLocalRows() || dLocalCols != result.numLocalRows()) {
        return;
    }
    
    result.setZero();
    
    for (int i = 0; i < dLocalCols; ++i) {
        double sum = 0.0;
        for (int j = 0; j < dLocalRows; ++j) {
            sum += item(j, i) * other[j];
        }
        result[i] = sum;
    }
    
    if (dDistributed && dCtx->getSize() > 1) {
        dCtx->allReduce(result.getData(), dLocalCols, MPI_SUM);
    }
}

matrix& matrix::transform(std::function<void(int rows, int cols, double* matrix)> transformer)
{
    transformer(dLocalRows, dLocalCols, dData.get());
    return *this;
}

void matrix::transform(matrix& result, std::function<void(int rows, int cols, double* matrix)> transformer) const
{
    if (result.dLocalRows != dLocalRows || result.dLocalCols != dLocalCols) {
        return;
    }
    result.dDistributed = dDistributed;
    transformer(dLocalRows, dLocalCols, result.dData.get());
}

void matrix::orthogonalize(bool doublePass, double zeroTol)
{
    int numPasses = doublePass ? 2 : 1;
    
    for (int col = 0; col < dLocalCols; ++col) {
        for (int pass = 0; pass < numPasses; ++pass) {
            for (int prevCol = 0; prevCol < col; ++prevCol) {
                double factor = 0.0;
                
                for (int i = 0; i < dLocalRows; ++i) {
                    factor += item(i, prevCol) * item(i, col);
                }
                
                if (dDistributed && dCtx->getSize() > 1) {
                    dCtx->allReduce(&factor, 1, MPI_SUM);
                }
                
                for (int i = 0; i < dLocalRows; ++i) {
                    item(i, col) -= factor * item(i, prevCol);
                }
            }
        }
        
        double norm = 0.0;
        for (int i = 0; i < dLocalRows; ++i) {
            norm += item(i, col) * item(i, col);
        }
        
        if (dDistributed && dCtx->getSize() > 1) {
            dCtx->allReduce(&norm, 1, MPI_SUM);
        }
        
        if (norm > zeroTol) {
            norm = 1.0 / std::sqrt(norm);
            for (int i = 0; i < dLocalRows; ++i) {
                item(i, col) *= norm;
            }
        }
    }
}

void matrix::orthogonalizeLast(int ncols, bool doublePass, double zeroTol)
{
    if (ncols == -1) {
        ncols = dLocalCols;
    }
    if (ncols <= 0 || ncols > dLocalCols) {
        return;
    }
    
    int lastCol = ncols - 1;
    int numPasses = doublePass ? 2 : 1;
    
    for (int pass = 0; pass < numPasses; ++pass) {
        for (int prevCol = 0; prevCol < lastCol; ++prevCol) {
            double factor = 0.0;
            
            for (int i = 0; i < dLocalRows; ++i) {
                factor += item(i, prevCol) * item(i, lastCol);
            }
            
            if (dDistributed && dCtx->getSize() > 1) {
                dCtx->allReduce(&factor, 1, MPI_SUM);
            }
            
            for (int i = 0; i < dLocalRows; ++i) {
                item(i, lastCol) -= factor * item(i, prevCol);
            }
        }
    }
    
    double norm = 0.0;
    for (int i = 0; i < dLocalRows; ++i) {
        norm += item(i, lastCol) * item(i, lastCol);
    }
    
    if (dDistributed && dCtx->getSize() > 1) {
        dCtx->allReduce(&norm, 1, MPI_SUM);
    }
    
    if (norm > zeroTol) {
        norm = 1.0 / std::sqrt(norm);
        for (int i = 0; i < dLocalRows; ++i) {
            item(i, lastCol) *= norm;
        }
    }
}

void matrix::inverse(matrix& result) const
{
    if (dDistributed) {
        return;
    }
    
    if (dLocalRows != dLocalCols || dLocalRows != result.dLocalRows || 
        dLocalCols != result.dLocalCols) {
        return;
    }
    
    int size = dLocalRows * dLocalCols;
    std::memcpy(result.dData.get(), dData.get(), size * sizeof(double));
    
    std::vector<double> identity(size, 0.0);
    for (int i = 0; i < dLocalRows; ++i) {
        identity[i * dLocalCols + i] = 1.0;
    }
    
    for (int i = 0; i < dLocalRows; ++i) {
        int pivot = i;
        double maxVal = std::abs(result.item(i, i));
        for (int k = i + 1; k < dLocalRows; ++k) {
            if (std::abs(result.item(k, i)) > maxVal) {
                maxVal = std::abs(result.item(k, i));
                pivot = k;
            }
        }
        
        if (maxVal < 1.0e-15) {
            return;
        }
        
        if (pivot != i) {
            for (int j = 0; j < dLocalCols; ++j) {
                std::swap(result.item(i, j), result.item(pivot, j));
                std::swap(identity[i * dLocalCols + j], identity[pivot * dLocalCols + j]);
            }
        }
        
        double pivotVal = result.item(i, i);
        for (int j = 0; j < dLocalCols; ++j) {
            result.item(i, j) /= pivotVal;
            identity[i * dLocalCols + j] /= pivotVal;
        }
        
        for (int k = 0; k < dLocalRows; ++k) {
            if (k != i) {
                double factor = result.item(k, i);
                for (int j = 0; j < dLocalCols; ++j) {
                    result.item(k, j) -= factor * result.item(i, j);
                    identity[k * dLocalCols + j] -= factor * identity[i * dLocalCols + j];
                }
            }
        }
    }
    
    std::memcpy(result.dData.get(), identity.data(), size * sizeof(double));
}

void matrix::inverse()
{
    matrix temp(*this);
    inverse(temp);
    *this = temp;
}

}
}

