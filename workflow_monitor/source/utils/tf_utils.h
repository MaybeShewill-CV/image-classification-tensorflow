// Copyright 2019 Baidu Inc. All Rights Reserved.
// Author: Luo Yao (luoyao@baidu.com)
// File: tf_utls.h
// Date: 2021/4/26 下午2:03

#ifndef WORKFLOW_MONITOR_TF_UTLS_H
#define WORKFLOW_MONITOR_TF_UTLS_H

#include <string>
#include <iostream>

#include <glog//logging.h>

#include "file_system_utils//file_system_processor.h"

namespace wf_monitor {
    namespace utils {

        using beec::common::file_system_utils::FileSystemProcessor;

        /***
         *
         * @param model_dir
         * @return
         */
        bool get_latest_checkpoint(const std::string& model_dir, std::string& model_checkpoint_path) {
            if (!FileSystemProcessor::is_directory_exist(model_dir)) {
                LOG(ERROR) << "Model dir: " << model_dir << ", not exist";
                return false;
            }
            // read checkpoint file
            std::string check_point_record_file_name = "checkpoint";
            std::string check_point_record_file_path = FileSystemProcessor::combine_path(
                    model_dir, check_point_record_file_name);
            if (!FileSystemProcessor::is_file_exist(check_point_record_file_path)) {
                LOG(ERROR) << "Check point file path: " << check_point_record_file_path << ", not exist";
                return false;
            }
            // read lasted checkpoint model name from checkpoint file
            std::ifstream checkpoint_file;
            checkpoint_file.open(check_point_record_file_path, std::ios::in);
            if (!checkpoint_file.is_open() || !checkpoint_file.good()) {
                LOG(ERROR) << "Open checkpoint file: " << check_point_record_file_path << " failed";
                return false;
            }
            const int char_size = 128;
            char checkpoint_model_name_cstr[char_size];
            checkpoint_file.getline(checkpoint_model_name_cstr, char_size);
            std::string checkpoint_model_name(checkpoint_model_name_cstr);
            checkpoint_model_name = checkpoint_model_name.substr(
                    checkpoint_model_name.find_first_of('\"') + 1,
                    checkpoint_model_name.find_last_of('\"') - checkpoint_model_name.find_first_of('\"') - 1
                    );
            // check index file path
            std::string index_file_name = checkpoint_model_name + ".index";
            std::string index_file_path = FileSystemProcessor::combine_path(model_dir, index_file_name);
            if (!FileSystemProcessor::is_file_exist(index_file_path)) {
                LOG(ERROR) << "Index file: " << index_file_path << ", not exist";
                return false;
            }
            // check meta file path
            std::string meta_file_name = checkpoint_model_name + ".meta";
            std::string meta_file_path = FileSystemProcessor::combine_path(model_dir, meta_file_name);
            if (!FileSystemProcessor::is_file_exist(meta_file_path)) {
                LOG(ERROR) << "Meta file: " << meta_file_path << ", not exist";
                return false;
            }
            // check data file path
            std::string data_file_name = checkpoint_model_name + ".data-00000-of-00001";
            std::string data_file_path = FileSystemProcessor::combine_path(model_dir, data_file_name);
            if (!FileSystemProcessor::is_file_exist(data_file_path)) {
                LOG(ERROR) << "Data file: " << data_file_path << ", not exist";
                return false;
            }
            checkpoint_file.close();

            model_checkpoint_path = FileSystemProcessor::combine_path(model_dir, checkpoint_model_name);
            return true;
        }


    }
}

#endif //WORKFLOW_MONITOR_TF_UTLS_H
