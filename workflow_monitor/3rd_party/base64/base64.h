/************************************************
* Copyright 2021 MaybeShewill-CV All Rights Reserved..
* Author: MaybeShewill-CV
* File: base64.h
* Date: 2019/10/18 下午5:12
************************************************/

#ifndef MNN_BASE64_H
#define MNN_BASE64_H

#include <string>

namespace beec {
namespace base64 {

class Base64 {
public:
    /***
     *
     */
    Base64() = delete;

    /***
     *
     */
    ~Base64() = default;

    /***
     * 赋值构造函数
     * @param transformer
     */
    Base64(const Base64 &transformer) = delete;

    /***
     * 复制构造函数
     * @param transformer
     * @return
     */
    Base64 &operator=(const Base64 &transformer) = delete;

    /***
     * Base64 encode string
     * @param len
     * @return
     */
    static std::string base64_encode(unsigned char const* , unsigned int len);

    /***
     * Base64 encode string
     * @param input
     * @return
     */
    static std::string base64_encode(const std::string& input);

    /***
     * Base64 decode string
     * @param s
     * @return
     */
    static std::string base64_decode(const std::string& s);
};

}
}

#endif //MNN_BASE64_H
