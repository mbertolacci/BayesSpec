#ifndef SRC_MPI_HPP_
#define SRC_MPI_HPP_

#include <RcppEigen.h>
#define OMPI_SKIP_MPICXX
#define MPICH_SKIP_MPICXX
#include <mpi.h>

namespace bayesspec {

class MPI {
public:
    static int rank() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }

    static int size() {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }

    static void broadcast(double& x, int senderRank) {
        MPI_Bcast(
            &x,
            1,
            MPI_DOUBLE,
            senderRank,
            MPI_COMM_WORLD
        );
    }

    static void broadcast(unsigned int& x, int senderRank) {
        MPI_Bcast(
            &x,
            1,
            MPI_UNSIGNED,
            senderRank,
            MPI_COMM_WORLD
        );
    }

    static void broadcast(Eigen::MatrixXd& x, int senderRank, bool resize = false) {
        if (resize) {
            int myRank = rank();
            unsigned int rows = 0;
            unsigned int cols = 0;
            if (myRank == senderRank) {
                rows = x.rows();
                cols = x.cols();
            }
            broadcast(rows, senderRank);
            broadcast(cols, senderRank);
            if (myRank != senderRank) {
                x.resize(rows, cols);
            }
        }
        MPI_Bcast(
            x.data(),
            x.size(),
            MPI_DOUBLE,
            senderRank,
            MPI_COMM_WORLD
        );
    }

    static void broadcast(Eigen::VectorXd& x, int senderRank) {
        MPI_Bcast(
            x.data(),
            x.size(),
            MPI_DOUBLE,
            senderRank,
            MPI_COMM_WORLD
        );
    }

    static void broadcast(Eigen::VectorXi& x, int senderRank) {
        MPI_Bcast(
            x.data(),
            x.size(),
            MPI_INT,
            senderRank,
            MPI_COMM_WORLD
        );
    }
};

}  // namespace bayesspec

#endif  // SRC_MPI_HPP_
