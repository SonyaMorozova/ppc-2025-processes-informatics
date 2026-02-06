#pragma once

#include <mpi.h>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "task/include/task.hpp"

namespace morozova_s_broadcast {

class MorozovaSBroadcastMPI : public BaseTask {
 public:
  static ppc::task::TypeOfTask GetStaticTypeOfTask();

  explicit MorozovaSBroadcastMPI(const InType &in);
  explicit MorozovaSBroadcastMPI(const InType &in, int root);

  [[nodiscard]] int GetRoot() const {
    return root_;
  }

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  static void CustomBroadcast(void *buffer, int count, MPI_Datatype type, int root, MPI_Comm comm);

  int root_;
};

}  // namespace morozova_s_broadcast
