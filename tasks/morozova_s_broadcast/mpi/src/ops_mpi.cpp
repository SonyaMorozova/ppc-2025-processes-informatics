#include "morozova_s_broadcast/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_broadcast {

ppc::task::TypeOfTask MorozovaSBroadcastMPI::GetStaticTypeOfTask() {
  return ppc::task::TypeOfTask::kMPI;
}

MorozovaSBroadcastMPI::MorozovaSBroadcastMPI(const InType &in) : root_(0) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

MorozovaSBroadcastMPI::MorozovaSBroadcastMPI(const InType &in, int root) : root_(root) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MorozovaSBroadcastMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  return root_ >= 0 && root_ < size;
}

bool MorozovaSBroadcastMPI::PreProcessingImpl() {
  return true;
}

bool MorozovaSBroadcastMPI::RunImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int data_size = 0;
  if (rank == root_) {
    data_size = static_cast<int>(GetInput().size());
  }

  CustomBroadcast(&data_size, 1, MPI_INT, root_, MPI_COMM_WORLD);

  GetOutput().resize(data_size);

  if (rank == root_ && data_size > 0) {
    std::copy(GetInput().begin(), GetInput().end(), GetOutput().begin());
  }

  if (data_size > 0) {
    CustomBroadcast(GetOutput().data(), data_size, MPI_INT, root_, MPI_COMM_WORLD);
  }

  return true;
}

bool MorozovaSBroadcastMPI::PostProcessingImpl() {
  return true;
}

void MorozovaSBroadcastMPI::CustomBroadcast(void *buffer, int count, MPI_Datatype type, int root, MPI_Comm comm) {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  int vrank = (rank - root + size) % size;

  for (int step = 1; step < size; step <<= 1) {
    if (vrank < step) {
      int dst = vrank + step;
      if (dst < size) {
        MPI_Send(buffer, count, type, (dst + root) % size, 0, comm);
      }
    } else if (vrank < 2 * step) {
      MPI_Recv(buffer, count, type, (vrank - step + root) % size, 0, comm, MPI_STATUS_IGNORE);
    }
  }
}

}  // namespace morozova_s_broadcast
