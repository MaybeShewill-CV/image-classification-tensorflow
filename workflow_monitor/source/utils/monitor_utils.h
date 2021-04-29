// Copyright 2021 MaybeShewill-CV All Rights Reserved..
// Author: MaybeShewill-CV
// File: tf_utls.h
// Date: 2021/4/26 下午2:03

#ifndef WORKFLOW_MONITOR_TF_UTLS_H
#define WORKFLOW_MONITOR_TF_UTLS_H

#include <string>

namespace wf_monitor {
namespace utils {

class MonitorUtils {
public:
    /***
     *
     */
    MonitorUtils() = delete;

    /***
     *
     */
    ~MonitorUtils() = default;

    /***
    * copy constructor
    * @param transformer
    */
    MonitorUtils(const MonitorUtils& transformer) = delete;

    /***
     * assign constructor
     * @param transformer
     * @return
     */
    MonitorUtils &operator=(const MonitorUtils& transformer) = delete;

public:

    /***
     * get latest checkpoint model file path
     * @param model_dir
     * @return
     */
    static bool get_latest_checkpoint_path(const std::string& model_dir, std::string& model_checkpoint_path);

    /***
     * check if the checkpoint model has been evaluated
     * @param eval_log_file_path
     * @param model_name
     * @return
     */
    static bool is_checkpoint_model_evaluated(
        const std::string& eval_log_file_path,
        const std::string& model_name);

    /***
    * judge if the training process is alive
    * @return
    */
    static bool is_net_training_process_alive();

    /***
     * judge if the evaluating process is alive
     * @return
     */
    static bool is_net_evaluating_process_alive();

    /***
     *
     * @param training_log_files
     * @return
     */
    static bool get_all_training_log_files(
        const std::string& log_dir,
        std::vector<std::string>& training_log_files);

    /***
     *
     * @param file_path
     * @return
     */
    static std::time_t get_file_last_modified_time(const std::string& file_path);

    /***
     *
     * @param log_dir
     * @param latest_log_file_path
     * @return
     */
    static bool get_latest_training_log_file(const std::string& log_dir, std::string& latested_log_file_path);

    /***
     *
     * @param log_dir
     * @param model_name
     * @return
     */
    static bool get_training_model_name(const std::string& log_dir, std::string& model_name);

    /***
     *
     * @param log_dir
     * @param dataset_name
     * @return
     */
    static bool get_training_dataset_name(const std::string& log_dir, std::string& dataset_name);

    /***
     *
     * @param project_base_dir
     * @param model_save_dir
     * @return
     */
    static bool get_checkpoint_model_save_dir(const std::string& project_base_dir, std::string& model_save_dir);

    /***
     *
     * @param log_dir
     * @param eval_log_file_path
     * @return
     */
    static bool get_eval_log_file_path(const std::string& log_dir, std::string& eval_log_file_path);

    /***
     *
     * @param project_dir
     * @param dataset_name
     * @param dataset_flag
     * @param image_count
     * @param precision
     * @param recall
     * @param f1
     * @return
     */
    static bool get_latest_checkpoint_model_eval_statics(
        const std::string& project_dir, std::string& dataset_name,
        std::string& dataset_flag,
        int32_t* image_count, float_t* precision, float_t* recall, float_t* f1);

    /***
     *
     * @param project_dir
     * @param epoch
     * @param train_loss
     * @param test_loss
     * @param train_acc
     * @param test_acc
     * @return
     */
    static bool get_model_training_statics(
        const std::string& project_dir, int* epoch, float* train_loss,
        float* test_loss, float* train_acc, float* test_acc);

    /***
     *
     * @param epoch
     * @return
     */
    static bool get_cur_train_epoch(int* epoch);

private:
    /***
     *
     * @param eval_log_file_path
     * @param checkpoint_model_name
     * @param dataset_name
     * @param dataset_flag
     * @param image_count
     * @param precision
     * @param recall
     * @param f1
     * @return
     */
    static bool _get_checkpoint_model_eval_statics_impl(
        const std::string& eval_log_file_path,
        std::string& dataset_name,
        std::string& dataset_flag,
        int32_t* image_count, float_t* precision, float_t* recall, float_t* f1);

    /***
     *
     * @param trainning_log_file_path
     * @param epoch
     * @param train_loss
     * @param test_loss
     * @param train_acc
     * @param test_acc
     * @return
     */
    static bool _get_model_training_statics_impl(
        const std::string& trainning_log_file_path,
        int* epoch, float* train_loss, float* test_loss, float* train_acc, float* test_acc);

    /***
     *
     * @param process_name
     * @return
     */
    static bool _is_process_alive(const std::string& process_name);
};
}
}

#include "monitor_utils.inl"

#endif //WORKFLOW_MONITOR_TF_UTLS_H
