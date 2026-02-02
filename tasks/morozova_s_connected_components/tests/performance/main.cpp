#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "morozova_s_connected_components/common/include/common.hpp"
#include "morozova_s_connected_components/mpi/include/ops_mpi.hpp"
#include "morozova_s_connected_components/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace morozova_s_connected_components {

class MorozovaSRunPerfTestConnectedComponents : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kImageSize_ = 100;
  InType input_data_;

  void SetUp() override {
    input_data_ = std::vector<std::vector<int>>(kImageSize_, std::vector<int>(kImageSize_, 0));
    for (int i = 10; i < 40; ++i) {
      input_data_[i][20] = 1;
      input_data_[i][21] = 1;
    }
    for (int j = 60; j < 90; ++j) {
      input_data_[50][j] = 1;
      input_data_[51][j] = 1;
    }
    for (int i = 70; i < 85; ++i) {
      for (int j = 70; j < 85; ++j) {
        input_data_[i][j] = 1;
      }
    }
    for (int i = 0; i < 20; ++i) {
      input_data_[80 + i][10 + i] = 1;
    }
    input_data_[5][5] = 1;
    input_data_[95][95] = 1;
    input_data_[10][90] = 1;
    input_data_[90][10] = 1;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }
    if (!output_data.empty() && output_data.back().size() == 1) {
      output_data.pop_back();
    }
    if (output_data.empty()) {
      return false;
    }
    int object_count = 0;
    int labeled_count = 0;
    for (std::size_t i = 0; i < input_data_.size(); ++i) {
      if (i >= output_data.size()) {
        return false;
      }
      for (std::size_t j = 0; j < input_data_[i].size(); ++j) {
        if (j >= output_data[i].size()) {
          return false;
        }
        if (input_data_[i][j] == 1) {
          object_count++;
          if (output_data[i][j] > 0) {
            labeled_count++;
          }
        } else {
          if (output_data[i][j] != 0) {
            return false;
          }
        }
      }
    }
    return object_count == labeled_count;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(MorozovaSRunPerfTestConnectedComponents, RunPerfModes) {
  ExecuteTest(GetParam());
}
const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, MorozovaSConnectedComponentsMPI, MorozovaSConnectedComponentsSEQ>(
        PPC_SETTINGS_morozova_s_connected_components);
const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = MorozovaSRunPerfTestConnectedComponents::CustomPerfTestName;
INSTANTIATE_TEST_SUITE_P(RunModeTests, MorozovaSRunPerfTestConnectedComponents, kGtestValues, kPerfTestName);

}  // namespace morozova_s_connected_components
