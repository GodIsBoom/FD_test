#include "fastdeploy/vision.h"
int main() {
  std::string model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel";
  std::string params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams";
  std::string infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml";
  // 模型推理的配置信息
  fastdeploy::RuntimeOption option;
  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file, infer_cfg_file, option);

  assert(model.Initialized()); // 判断模型是否初始化成功

  cv::Mat im = cv::imread("test_det.jpg");
  fastdeploy::vision::DetectionResult result;
  
  assert(model.Predict(&im, &result)); // 判断是否预测成功

  std::cout << result.Str() << std::endl;

  cv::Mat vis_im = fastdeploy::vision::Visualize::VisDetection(im, result, 0.5);
  // 可视化结果保存到本地
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result save in vis_result.jpg" << std::endl;
  return 0;
}