#ifndef INCLUDED_MATRIX_H
#define INCLUDED_MATRIX_H

#include <memory>
#include <functional>
#include "parallelContext.hpp"
#include "vector.hpp"

namespace shonClooud::shonTA::rom::framework {
namespace core {

class parallelContext;
class vector;

class matrix
{
public:
    matrix(int globalRows,
           int globalCols,
           const parallelContext& ctx,
           bool distributed = true);
    matrix(double* data,
           int localRows,
           int localCols,
           int globalRows,
           int globalCols,
           const parallelContext& ctx,
           bool distributed = true,
           bool copyData = true);
    matrix(const matrix& other);
    matrix(matrix&& other) noexcept;
    ~matrix();
    matrix& operator=(const matrix& rhs);
    matrix& operator=(matrix&& rhs) noexcept;

    [[nodiscard]] bool isDistributed() const noexcept { return dDistributed; }
    [[nodiscard]] int numLocalRows() const noexcept { return dLocalRows; }
    [[nodiscard]] int numLocalCols() const noexcept { return dLocalCols; }
    [[nodiscard]] int numGlobalRows() const;
    [[nodiscard]] int numGlobalCols() const noexcept { return dGlobalCols; }
    
    double& item(int i, int j) { return dData.get()[i * dLocalCols + j]; }
    const double& item(int i, int j) const { return dData.get()[i * dLocalCols + j]; }
    
    [[nodiscard]] double* getData() noexcept { return dData.get(); }
    [[nodiscard]] const double* getData() const noexcept { return dData.get(); }
    
    [[nodiscard]] double norm() const;
    [[nodiscard]] double frobeniusNorm() const;
    void scale(double alpha);
    void setZero();
    
    void matVecMult(const vector& x, vector& y) const;
    void matVecMultAdd(const vector& x, vector& y, double alpha = 1.0) const;
    
    void mult(const matrix& other, matrix& result) const;
    
    void transposeMult(const matrix& other, matrix& result) const;
    void transposeMult(const vector& other, vector& result) const;
    
    matrix& transform(std::function<void(int rows, int cols, double* matrix)> transformer);
    void transform(matrix& result, std::function<void(int rows, int cols, double* matrix)> transformer) const;
    
    void orthogonalize(bool doublePass = false, double zeroTol = 1.0e-15);
    void orthogonalizeLast(int ncols = -1, bool doublePass = false, double zeroTol = 1.0e-15);
    
    void inverse(matrix& result) const;
    void inverse();
    
    [[nodiscard]] const parallelContext& getContext() const noexcept { return *dCtx; }

private:
    std::shared_ptr<const parallelContext> dCtx;
    std::shared_ptr<double> dData;
    int dLocalRows;  
    int dLocalCols;
    int dGlobalRows;
    int dGlobalCols;
    bool dDistributed;
};

}
}

#endif

