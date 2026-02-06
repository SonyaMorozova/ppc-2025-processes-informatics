#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>

#include "morozova_s_broadcast/common/include/common.hpp"
#include "morozova_s_broadcast/mpi/include/ops_mpi.hpp"
#include "morozova_s_broadcast/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace morozova_s_broadcast {

class MorozovaSBroadcastFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param) + "_" + std::to_string(std::get<0>(test_param));
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    const int insz = std::get<0>(params);
    input_data_.resize(insz);
    for (int i = 0; i < insz; ++i) {
      input_data_[i] = i;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == input_data_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

namespace {

TEST_P(MorozovaSBroadcastFuncTests, BroadcastTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 4> kTestParam = {std::make_tuple(1, "size_1"), std::make_tuple(2, "size_2"),
                                            std::make_tuple(10, "size_10"), std::make_tuple(100, "size_100")};
const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<MorozovaSBroadcastMPI, InType>(kTestParam, PPC_SETTINGS_morozova_s_broadcast),
    ppc::util::AddFuncTask<MorozovaSBroadcastSEQ, InType>(kTestParam, PPC_SETTINGS_morozova_s_broadcast));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);
const auto kTestName = MorozovaSBroadcastFuncTests::PrintFuncTestName<MorozovaSBroadcastFuncTests>;

INSTANTIATE_TEST_SUITE_P(BroadcastTests, MorozovaSBroadcastFuncTests, kGtestValues, kTestName);

}  // namespace

}  // namespace morozova_s_broadcast
