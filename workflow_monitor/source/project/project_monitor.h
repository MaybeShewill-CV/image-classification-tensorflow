// Copyright 2021 MaybeShewill-CV All Rights Reserved..
// Author: MaybeShewill-CV
// File: project_monitor.h
// Date: 2021/4/29 上午1:47

#ifndef WORKFLOW_MONITOR_PROJECTMONITOR_H
#define WORKFLOW_MONITOR_PROJECTMONITOR_H

#include "toml/toml.hpp"
#include "rapidjson/document.h"
#include "rapidjson/writer.h"

namespace wf_monitor {
    namespace project {

        struct TrainStatic {
            int epoch,
            float training_loss;
            float testing_loss;
            float training_accuracy;
            float testing_accuracy;

            /***
             * convert to json doc
             * @return
             */
            rapidjson::Document to_json() const {
                rapidjson::Document doc;
                doc.SetObject();
                auto&& allocator = doc.GetAllocator();

                // record epoch
                rapidjson::Value epoch_json;
                epoch_json.SetInt(epoch);
                doc.AddMember("epoch", epoch_json, allocator);
                // record train loss
                rapidjson::Value train_loss_json;
                train_loss_json.SetFloat(training_loss);
                doc.AddMember("train_loss", train_loss_json, allocator);
                // record test loss
                rapidjson::Value test_loss_json;
                test_loss_json.SetFloat(testing_loss);
                doc.AddMember("test_loss", test_loss_json, allocator);
                // record train accuracy
                rapidjson::Value train_acc_json;
                train_acc_json.SetFloat(training_accuracy);
                doc.AddMember("train_accuracy", train_acc_json, allocator);
                // record test accuracy
                rapidjson::Value test_acc_json;
                test_acc_json.SetFloat(testing_loss);
                doc.AddMember("test_accuracy", test_acc_json, allocator);

                return doc;
            }

            /***
             * convert to string
             * @return
             */
            std::string to_str() {
                auto json_doc = to_json();
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                json_doc.Accept(writer);

                return buffer.GetString();
            }
        };

        struct EvalStatic {
            std::string dataset_name;
            std::string dataset_flag;
            int image_count;
            float precision;
            float recall;
            float f1;

            /***
             * convert to json doc
             * @return
             */
            rapidjson::Document to_json() const {
                rapidjson::Document doc;
                doc.SetObject();
                auto&& allocator = doc.GetAllocator();

                // record dataset name
                rapidjson::Value dataset_name_json;
                dataset_name_json.SetString(dataset_name.c_str(), dataset_name.size(), allocator);
                doc.AddMember("dataset_name", dataset_name_json, allocator);
                // record dataset flag
                rapidjson::Value dataset_flag_json;
                dataset_flag_json.SetFloat(dataset_flag.c_str(), dataset_flag.size(), allocator);
                doc.AddMember("dataset_flag", dataset_flag_json, allocator);
                // record image count
                rapidjson::Value image_count_json;
                image_count_json.SetInt(image_count);
                doc.AddMember("image_count", image_count_json, allocator);
                // record eval precision
                rapidjson::Value precision_json;
                precision_json.SetFloat(precision);
                doc.AddMember("eval_precision", precision_json, allocator);
                // record eval recall
                rapidjson::Value recall_json;
                recall_json.SetFloat(recall);
                doc.AddMember("eval_recall", recall_json, allocator);
                // record eval f1
                rapidjson::Value f1_json;
                f1_json.SetFloat(f1);
                doc.AddMember("eval_f1", f1_json, allocator);

                return doc;
            }

            /***
             * convert to str
             * @return
             */
            std::string to_str() {
                auto json_doc = to_json();
                rapidjson::StringBuffer buffer;
                rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                json_doc.Accept(writer);

                return buffer.GetString();
            }

        };

        class ProjectMonitor {
        public:
            /***
             * Default constructor
             */
            ProjectMonitor() = default;

            /***
             * Default destroyer
             */
            ~ProjectMonitor() = default;

            /***
             *
             * @param config
             */
            explicit ProjectMonitor(const toml::value& config);

            /***
            * copy constructor
            * @param transformer
            */
            ProjectMonitor(const FaceDetector& transformer) = delete;

            /***
             * assign constructor
             * @param transformer
             * @return
             */
            ProjectMonitor &operator=(const FaceDetector& transformer) = delete;

        };
    }
}

#endif //WORKFLOW_MONITOR_PROJECTMONITOR_H
