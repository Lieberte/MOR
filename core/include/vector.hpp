#ifndef INCLUDED_VECTOR_H
#define INCLUDED_VECTOR_H

#include <memory>

namespace shonClooud::shonTA::rom::framework {
namespace core {

class ParallelContext;

class Vector
{
public:
    Vector(int global_size,
           const ParallelContext& ctx,
           bool distributed = true);
    Vector(double* data,
           int local_size,
           const ParallelContext& ctx,
           bool distributed = true,
           bool copy_data = true);
    Vector(const Vector& other);
    Vector(Vector&& other) noexcept;
    ~Vector();
    Vector& operator=(const Vector& rhs);
    Vector& operator=(Vector&& rhs) noexcept;

    [[nodiscard]] bool isDistributed() const noexcept { return d_distributed; }
    [[nodiscard]] int numLocalRows() const noexcept { return d_local_size; }
    [[nodiscard]] int numGlobalRows() const;
    
    double& item(int i) { return d_data.get()[i]; }
    const double& item(int i) const { return d_data.get()[i]; }
    double& operator[](int i) { return d_data.get()[i]; }
    const double& operator[](int i) const { return d_data.get()[i]; }
    
    [[nodiscard]] double* getData() noexcept { return d_data.get(); }
    [[nodiscard]] const double* getData() const noexcept { return d_data.get(); }
    
    [[nodiscard]] double norm() const;
    [[nodiscard]] double dot(const Vector& other) const;
    void scale(double alpha);
    void setZero();
    
    [[nodiscard]] const ParallelContext& getContext() const noexcept { return *d_ctx; }

private:
    std::shared_ptr<const ParallelContext> d_ctx;
    std::shared_ptr<double> d_data;
    int d_local_size;
    int d_global_size;
    bool d_distributed;
};

}
}

#endif
