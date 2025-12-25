# ifndef included_ParallelContext_h
# define included_ParallelContext_h

#include "mpi.h"
#include <vector>

namespace shonClooud::shonTA::rom::framework::core {

class parallelContext
{
public:
    parallelContext(MPI_Comm comm = MPI_COMM_WORLD);
    ~parallelContext();

    int getRank() const { return dRank; }
    int getSize() const { return dSize; }
    MPI_Comm getComm() const { return dComm; }

    int splitDimension(int globalDim) const;
    void getGlobalOffsets(int localDim, std::vector<int>& offsets) const;

    void allReduce(double* data, int count, MPI_Op op) const;
    void allGather(const void* sendbuf, void* recvbuf, int count,
                    MPI_Datatype type) const;
    void broadcast(void* data, int count, int root, MPI_Datatype type) const;
    void barrier() const;

private:
    MPI_Comm dComm;
    int dRank;
    int dSize;
    bool dOwnsComm;
};

}

#endif