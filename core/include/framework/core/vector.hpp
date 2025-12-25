#ifndef INCLUDED_VECTOR_H
#define INCLUDED_VECTOR_H

#include <memory>
#include <functional>
#include <vector>

namespace shonClooud::shonTA::rom::framework {
namespace core {

class parallelContext;

class vector
{
public:
    vector(int globalSize,
           const parallelContext& ctx,
           bool distributed = true);
    vector(double* data,
           int localSize,
           const parallelContext& ctx,
           bool distributed = true,
           bool copyData = true);
    vector(const vector& other);
    vector(vector&& other) noexcept;
    ~vector();
    vector& operator=(const vector& rhs);
    vector& operator=(vector&& rhs) noexcept;

    [[nodiscard]] bool isDistributed() const noexcept { return dDistributed; }
    [[nodiscard]] int numLocalRows() const noexcept { return dLocalSize; }
    [[nodiscard]] int numGlobalRows() const;
    
    double& item(int i) { return dData.get()[i]; }
    const double& item(int i) const { return dData.get()[i]; }
    double& operator[](int i) { return dData.get()[i]; }
    const double& operator[](int i) const { return dData.get()[i]; }
    
    [[nodiscard]] double* getData() noexcept { return dData.get(); }
    [[nodiscard]] const double* getData() const noexcept { return dData.get(); }
    
    [[nodiscard]] double norm() const;
    [[nodiscard]] double norm2() const;
    [[nodiscard]] double dot(const vector& other) const;
    void scale(double alpha);
    void setZero();
    
    vector& transform(std::function<void(int size, double* vector)> transformer);
    void transform(vector& result, std::function<void(int size, double* vector)> transformer) const;
    
    double normalize();
    
    void plus(const vector& other, vector& result) const;
    void minus(const vector& other, vector& result) const;
    void mult(double factor, vector& result) const;
    void plusEqAx(double factor, const vector& other);
    
    vector& operator+=(const vector& other);
    vector& operator-=(const vector& other);
    
    [[nodiscard]] double max() const;
    [[nodiscard]] double min() const;
    [[nodiscard]] std::vector<int> maxN(int n) const;
    [[nodiscard]] std::vector<int> minN(int n) const;
    
    [[nodiscard]] double mean() const;
    
    [[nodiscard]] int findNearest(double value) const;
    [[nodiscard]] const parallelContext& getContext() const noexcept { return *dCtx; }

private:
    std::shared_ptr<const parallelContext> dCtx;
    std::shared_ptr<double> dData;
    int dLocalSize;
    int dGlobalSize;
    bool dDistributed;
};

}
}

#endif
