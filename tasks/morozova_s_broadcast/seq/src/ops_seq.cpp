#include "morozova_s_broadcast/seq/include/ops_seq.hpp"

#include "morozova_s_broadcast/common/include/common.hpp"

namespace morozova_s_broadcast {

MorozovaSBroadcastSEQ::MorozovaSBroadcastSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool MorozovaSBroadcastSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool MorozovaSBroadcastSEQ::PreProcessingImpl() {
  return true;
}

bool MorozovaSBroadcastSEQ::RunImpl() {
  GetOutput() = GetInput();
  return true;
}

bool MorozovaSBroadcastSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace morozova_s_broadcast
