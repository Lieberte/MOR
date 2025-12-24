# ifndef included_ParallelContext_h
# define included_ParallelContext_h

# include"mpi.h"
# include<vector>

namespace shonClooud { namespace shonTA { namespace rom { namespace framework { namespace core {

class ParallelContext
{
public:
    ParallelContext(MPI_Comm comm = MPI_COMM_WORLD);
    ~ParallelContext();

    int getRank() const { return d_rank; }
    int getSize() const { return d_size; }
    MPI_Comm getComm() const { return d_comm; }

    int splitDimension(int global_dim) const;
    void getGlobalOffsets(int local_dim, std::vector<int>& offsets) const;

    void allReduce(double* data, int count, MPI_Op op) const;
    void allGather(const void* sendbuf, void* recvbuf, int count,
                    MPI_Datatype type) const;
    void broadcast(void* data, int count, int root, MPI_Datatype type) const;
    void barrier() const;

private:
    MPI_Comm d_comm;
    int d_rank;
    int d_size;
    bool d_owns_comm;
};

} } } } }

#endif