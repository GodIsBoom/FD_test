#include "fastdeploy/vision.h"
int main() {
   std::string model_file = "ppyoloe_crn_l_300e_coco/model.pdmodel";
  std::string params_file = "ppyoloe_crn_l_300e_coco/model.pdiparams";
  std::string infer_cfg_file = "ppyoloe_crn_l_300e_coco/infer_cfg.yml";
  
  fastdeploy::RuntimeOption option; // Configuration information for model inference
  auto model = fastdeploy::vision::detection::PPYOLOE(model_file, params_file, infer_cfg_file, option);

  assert(model.Initialized()); // Determine if the model is initialized successfully

  cv::mat im = cv::imread("000000014439.jpg");
  fastdeploy::vision::DetectionResult result;
  
  assert(model.Predict(im)); // Determine whether the prediction is successful

  std::cout << result << std::endl;

  cv::mat vis_im = fastdeploy::vision::Visualize::VisDetection(im, result, 0.5);
  
  cv::imwrite("vis_result.jpg", vis_im); // The visualization results are saved locally

  return 0;
}