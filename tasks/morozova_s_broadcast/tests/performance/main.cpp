#include <gtest/gtest.h>

#include <cstring>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "morozova_s_broadcast/mpi/include/ops_mpi.hpp"
#include "morozova_s_broadcast/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozova_s_broadcast {

class MorozovaSRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 10000000;
  InType input_data_;

  void SetUp() override {
    input_data_.resize(kCount_, 1);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == input_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MorozovaSRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, MorozovaSBroadcastMPI, MorozovaSBroadcastSEQ>(
    PPC_SETTINGS_morozova_s_broadcast);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = MorozovaSRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovaSRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace morozova_s_broadcast
