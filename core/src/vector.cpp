#include "vector.hpp"
#include "parallelContext.hpp"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>
#include <limits>
#include <mpi.h>

namespace shonClooud::shonTA::rom::framework {
namespace core {

vector::vector(int globalSize,
               const parallelContext& ctx,
               bool distributed)
    : dCtx(std::make_shared<const parallelContext>(ctx))
    , dLocalSize(0)
    , dGlobalSize(globalSize)
    , dDistributed(distributed)
{
    if (dDistributed) {
        dLocalSize = dCtx->splitDimension(globalSize);
    } else {
        dLocalSize = globalSize;
    }
    
    double* rawData = new double[dLocalSize];
    std::fill_n(rawData, dLocalSize, 0.0);
    dData = std::shared_ptr<double>(rawData, [](double* p) { delete[] p; });
}

vector::vector(double* data,
               int localSize,
               const parallelContext& ctx,
               bool distributed,
               bool copyData)
    : dCtx(std::make_shared<const parallelContext>(ctx))
    , dLocalSize(localSize)
    , dGlobalSize(0)
    , dDistributed(distributed)
{
    if (copyData) {
        double* rawData = new double[localSize];
        std::memcpy(rawData, data, localSize * sizeof(double));
        dData = std::shared_ptr<double>(rawData, [](double* p) { delete[] p; });
    } else {
        dData = std::shared_ptr<double>(data, [](double*) {});
    }
    
    if (dDistributed) {
        std::vector<int> offsets;
        dCtx->getGlobalOffsets(localSize, offsets);
        dGlobalSize = offsets.back() + localSize;
    } else {
        dGlobalSize = localSize;
    }
}

vector::vector(const vector& other)
    : dCtx(other.dCtx)
    , dLocalSize(other.dLocalSize)
    , dGlobalSize(other.dGlobalSize)
    , dDistributed(other.dDistributed)
{
    double* rawData = new double[dLocalSize];
    std::memcpy(rawData, other.dData.get(), dLocalSize * sizeof(double));
    dData = std::shared_ptr<double>(rawData, [](double* p) { delete[] p; });
}

vector::vector(vector&& other) noexcept
    : dCtx(std::move(other.dCtx))
    , dData(std::move(other.dData))
    , dLocalSize(other.dLocalSize)
    , dGlobalSize(other.dGlobalSize)
    , dDistributed(other.dDistributed)
{
    other.dLocalSize = 0;
    other.dGlobalSize = 0;
    other.dDistributed = false;
}

vector::~vector() = default;

vector& vector::operator=(const vector& rhs)
{
    if (this != &rhs) {
        dCtx = rhs.dCtx;
        dLocalSize = rhs.dLocalSize;
        dGlobalSize = rhs.dGlobalSize;
        dDistributed = rhs.dDistributed;
        
        double* rawData = new double[dLocalSize];
        std::memcpy(rawData, rhs.dData.get(), dLocalSize * sizeof(double));
        dData = std::shared_ptr<double>(rawData, [](double* p) { delete[] p; });
    }
    return *this;
}

vector& vector::operator=(vector&& rhs) noexcept
{
    if (this != &rhs) {
        dCtx = std::move(rhs.dCtx);
        dData = std::move(rhs.dData);
        dLocalSize = rhs.dLocalSize;
        dGlobalSize = rhs.dGlobalSize;
        dDistributed = rhs.dDistributed;
        
        rhs.dLocalSize = 0;
        rhs.dGlobalSize = 0;
        rhs.dDistributed = false;
    }
    return *this;
}

int vector::numGlobalRows() const
{
    if (dDistributed && dGlobalSize == 0) {
        std::vector<int> offsets;
        dCtx->getGlobalOffsets(dLocalSize, offsets);
        if (!offsets.empty()) {
            int localGlobalSize = offsets.back() + dLocalSize;
            return localGlobalSize;
        }
    }
    return dGlobalSize;
}

double vector::norm() const
{
    double localSum = 0.0;
    for (int i = 0; i < dLocalSize; ++i) {
        localSum += dData.get()[i] * dData.get()[i];
    }
    
    if (dDistributed) {
        dCtx->allReduce(&localSum, 1, MPI_SUM);
    }
    
    return std::sqrt(localSum);
}

double vector::dot(const vector& other) const
{
    if (dLocalSize != other.dLocalSize) {
        return 0.0;
    }
    
    double localDot = 0.0;
    for (int i = 0; i < dLocalSize; ++i) {
        localDot += dData.get()[i] * other.dData.get()[i];
    }
    
    if (dDistributed) {
        dCtx->allReduce(&localDot, 1, MPI_SUM);
    }
    
    return localDot;
}

void vector::scale(double alpha)
{
    for (int i = 0; i < dLocalSize; ++i) {
        dData.get()[i] *= alpha;
    }
}

    void vector::setZero()
{
    std::fill_n(dData.get(), dLocalSize, 0.0);
}

double vector::norm2() const
{
    double localSum = 0.0;
    for (int i = 0; i < dLocalSize; ++i) {
        localSum += dData.get()[i] * dData.get()[i];
    }
    
    if (dDistributed) {
        dCtx->allReduce(&localSum, 1, MPI_SUM);
    }
    
    return localSum;
}

double vector::normalize()
{
    double normVal = norm();
    if (normVal > 1.0e-15) {
        scale(1.0 / normVal);
    }
    return normVal;
}

vector& vector::transform(std::function<void(int size, double* vector)> transformer)
{
    transformer(dLocalSize, dData.get());
    return *this;
}

void vector::transform(vector& result, std::function<void(int size, double* vector)> transformer) const
{
    if (result.dLocalSize != dLocalSize) {
        return;
    }
    result.dDistributed = dDistributed;
    transformer(dLocalSize, result.dData.get());
}

void vector::plus(const vector& other, vector& result) const
{
    if (dLocalSize != other.dLocalSize || dLocalSize != result.dLocalSize) {
        return;
    }
    
    for (int i = 0; i < dLocalSize; ++i) {
        result.dData.get()[i] = dData.get()[i] + other.dData.get()[i];
    }
}

void vector::minus(const vector& other, vector& result) const
{
    if (dLocalSize != other.dLocalSize || dLocalSize != result.dLocalSize) {
        return;
    }
    
    for (int i = 0; i < dLocalSize; ++i) {
        result.dData.get()[i] = dData.get()[i] - other.dData.get()[i];
    }
}

void vector::mult(double factor, vector& result) const
{
    if (dLocalSize != result.dLocalSize) {
        return;
    }
    
    for (int i = 0; i < dLocalSize; ++i) {
        result.dData.get()[i] = factor * dData.get()[i];
    }
}

    void vector::plusEqAx(double factor, const vector& other)
{
    if (dLocalSize != other.dLocalSize) {
        return;
    }
    
    for (int i = 0; i < dLocalSize; ++i) {
        dData.get()[i] += factor * other.dData.get()[i];
    }
}

vector& vector::operator+=(const vector& other)
{
    if (dLocalSize != other.dLocalSize) {
        return *this;
    }
    
    for (int i = 0; i < dLocalSize; ++i) {
        dData.get()[i] += other.dData.get()[i];
    }
    
    return *this;
}

vector& vector::operator-=(const vector& other)
{
    if (dLocalSize != other.dLocalSize) {
        return *this;
    }
    
    for (int i = 0; i < dLocalSize; ++i) {
        dData.get()[i] -= other.dData.get()[i];
    }
    
    return *this;
}

double vector::max() const
{
    if (dLocalSize == 0) {
        return std::numeric_limits<double>::lowest();
    }
    
    double localMax = *std::max_element(dData.get(), dData.get() + dLocalSize);
    
    if (dDistributed) {
        dCtx->allReduce(&localMax, 1, MPI_MAX);
    }
    
    return localMax;
}

double vector::min() const
{
    if (dLocalSize == 0) {
        return std::numeric_limits<double>::max();
    }
    
    double localMin = *std::min_element(dData.get(), dData.get() + dLocalSize);
    
    if (dDistributed) {
        dCtx->allReduce(&localMin, 1, MPI_MIN);
    }
    
    return localMin;
}

std::vector<int> vector::maxN(int n) const
{
    if (n <= 0 || dLocalSize == 0) {
        return std::vector<int>();
    }
    
    // Create vector of pairs (value, local_index)
    std::vector<std::pair<double, int>> localPairs;
    localPairs.reserve(dLocalSize);
    
    for (int i = 0; i < dLocalSize; ++i) {
        localPairs.emplace_back(dData.get()[i], i);
    }
    
    // Sort by value descending
    std::sort(localPairs.begin(), localPairs.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });
    
    // Get top n local indices
    int localN = std::min(n, dLocalSize);
    
    if (!dDistributed) {
        // Non-distributed: just return local indices
        std::vector<int> result;
        result.reserve(localN);
        for (int i = 0; i < localN; ++i) {
            result.push_back(localPairs[i].second);
        }
        return result;
    }
    
    int rank = dCtx->getRank();
    int size = dCtx->getSize();
    
    std::vector<double> sendValues(n);
    std::vector<int> sendIndices(n);
    double minValue = std::numeric_limits<double>::lowest();
    
    for (int i = 0; i < n; ++i) {
        if (i < localN) {
            sendValues[i] = localPairs[i].first;
            sendIndices[i] = localPairs[i].second;
        } else {
            sendValues[i] = minValue;
            sendIndices[i] = -1;
        }
    }
    
    std::vector<double> allValues(n * size);
    std::vector<int> allIndices(n * size);
    
    dCtx->allGather(sendValues.data(), allValues.data(), n, MPI_DOUBLE);
    dCtx->allGather(sendIndices.data(), allIndices.data(), n, MPI_INT);
    
    std::vector<int> globalOffsets;
    dCtx->getGlobalOffsets(dLocalSize, globalOffsets);
    
    std::vector<std::pair<double, int>> allPairs;
    allPairs.reserve(n * size);
    
    for (int proc = 0; proc < size; ++proc) {
        int procBaseOffset = globalOffsets.empty() ? 0 : globalOffsets[proc];
        for (int i = 0; i < n; ++i) {
            int idx = proc * n + i;
            if (allIndices[idx] >= 0) {
                int globalIdx = procBaseOffset + allIndices[idx];
                allPairs.emplace_back(allValues[idx], globalIdx);
            }
        }
    }
    
    std::sort(allPairs.begin(), allPairs.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first > b.first;
              });
    
    int finalN = std::min(n, static_cast<int>(allPairs.size()));
    std::vector<int> result;
    result.reserve(finalN);
    for (int i = 0; i < finalN; ++i) {
        result.push_back(allPairs[i].second);
    }
    
    return result;
}

std::vector<int> vector::minN(int n) const
{
    if (n <= 0 || dLocalSize == 0) {
        return std::vector<int>();
    }
    
    std::vector<std::pair<double, int>> localPairs;
    localPairs.reserve(dLocalSize);
    
    for (int i = 0; i < dLocalSize; ++i) {
        localPairs.emplace_back(dData.get()[i], i);
    }
    
    std::sort(localPairs.begin(), localPairs.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first < b.first;
              });
    
    int localN = std::min(n, dLocalSize);
    
    if (!dDistributed) {
        std::vector<int> result;
        result.reserve(localN);
        for (int i = 0; i < localN; ++i) {
            result.push_back(localPairs[i].second);
        }
        return result;
    }
    
    int rank = dCtx->getRank();
    int size = dCtx->getSize();
    
    std::vector<double> sendValues(n);
    std::vector<int> sendIndices(n);
    double maxValue = std::numeric_limits<double>::max();
    
    for (int i = 0; i < n; ++i) {
        if (i < localN) {
            sendValues[i] = localPairs[i].first;
            sendIndices[i] = localPairs[i].second;
        } else {
            sendValues[i] = maxValue;
            sendIndices[i] = -1;
        }
    }
    
    std::vector<double> allValues(n * size);
    std::vector<int> allIndices(n * size);
    
    dCtx->allGather(sendValues.data(), allValues.data(), n, MPI_DOUBLE);
    dCtx->allGather(sendIndices.data(), allIndices.data(), n, MPI_INT);
    
    std::vector<int> globalOffsets;
    dCtx->getGlobalOffsets(dLocalSize, globalOffsets);
    
    std::vector<std::pair<double, int>> allPairs;
    allPairs.reserve(n * size);
    
    for (int proc = 0; proc < size; ++proc) {
        int procBaseOffset = globalOffsets.empty() ? 0 : globalOffsets[proc];
        for (int i = 0; i < n; ++i) {
            int idx = proc * n + i;
            if (allIndices[idx] >= 0) {
                int globalIdx = procBaseOffset + allIndices[idx];
                allPairs.emplace_back(allValues[idx], globalIdx);
            }
        }
    }
    
    std::sort(allPairs.begin(), allPairs.end(),
              [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                  return a.first < b.first;
              });
    
    int finalN = std::min(n, static_cast<int>(allPairs.size()));
    std::vector<int> result;
    result.reserve(finalN);
    for (int i = 0; i < finalN; ++i) {
        result.push_back(allPairs[i].second);
    }
    
    return result;
}

double vector::mean() const
{
    if (dLocalSize == 0) {
        return 0.0;
    }
    
    double localSum = 0.0;
    for (int i = 0; i < dLocalSize; ++i) {
        localSum += dData.get()[i];
    }
    
    int totalSize = dLocalSize;
    if (dDistributed) {
        dCtx->allReduce(&localSum, 1, MPI_SUM);
        
        int localSizeInt = dLocalSize;
        dCtx->allReduce(&localSizeInt, 1, MPI_SUM);
        totalSize = localSizeInt;
    }
    
    return totalSize > 0 ? localSum / totalSize : 0.0;
}

int vector::findNearest(double value) const
{
    if (dLocalSize == 0) {
        return -1;
    }
    
    int localNearestIdx = 0;
    double minDist = std::abs(dData.get()[0] - value);
    
    for (int i = 1; i < dLocalSize; ++i) {
        double dist = std::abs(dData.get()[i] - value);
        if (dist < minDist) {
            minDist = dist;
            localNearestIdx = i;
        }
    }
    
    double localNearestValue = dData.get()[localNearestIdx];
    
    if (!dDistributed) {
        return localNearestIdx;
    }
    
    int rank = dCtx->getRank();
    int size = dCtx->getSize();
    
    std::vector<double> nearestDists(size);
    dCtx->allGather(&minDist, nearestDists.data(), 1, MPI_DOUBLE);
    
    int globalNearestProc = 0;
    double globalMinDist = nearestDists[0];
    for (int i = 1; i < size; ++i) {
        if (nearestDists[i] < globalMinDist) {
            globalMinDist = nearestDists[i];
            globalNearestProc = i;
        }
    }
    
    int nearestLocalIdx = localNearestIdx;
    dCtx->broadcast(&nearestLocalIdx, 1, MPI_INT, globalNearestProc);
    
    std::vector<int> globalOffsets;
    dCtx->getGlobalOffsets(dLocalSize, globalOffsets);
    int baseOffset = globalOffsets.empty() ? 0 : globalOffsets[globalNearestProc];
    
    return baseOffset + nearestLocalIdx;
}

}
}

