/************************************************
* Copyright 2019 Baidu Inc. All Rights Reserved.
* Author: Luo Yao (luoyao@baidu.com)
* File: base64.cpp
* Date: 2019/10/18 下午5:12
************************************************/

#include "base64.h"

#include <iostream>

namespace beec {
namespace base64 {

static const std::string base64_chars =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";


static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

inline size_t encode_len(size_t input_len) {
    return (input_len + 2) / 3 * 4;
}

void encode_char_array(unsigned char *encode_block, const unsigned char *decode_block) {
    encode_block[0] = (decode_block[0] & 0xfc) >> 2;
    encode_block[1] = ((decode_block[0] & 0x03) << 4) + ((decode_block[1] & 0xf0) >> 4);
    encode_block[2] = ((decode_block[1] & 0x0f) << 2) + ((decode_block[2] & 0xc0) >> 6);
    encode_block[3] = decode_block[2] & 0x3f;
}

void decode_char_array(unsigned char *encode_block, unsigned char *decode_block) {
    for (int i = 0; i < 4; ++i) {
        encode_block[i] = base64_chars.find(encode_block[i]);
    }
    decode_block[0] = (encode_block[0] << 2) + ((encode_block[1] & 0x30) >> 4);
    decode_block[1] = ((encode_block[1] & 0xf) << 4) + ((encode_block[2] & 0x3c) >> 2);
    decode_block[2] = ((encode_block[2] & 0x3) << 6) + encode_block[3];
}

/***
 * Base64 encode
 * @param bytes_to_encode
 * @param in_len
 * @return
 */
std::string Base64::base64_encode(unsigned char const *bytes_to_encode, unsigned int in_len) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];

    while (in_len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++) {
                ret += base64_chars[char_array_4[i]];
            }
            i = 0;
        }
    }

    if (i) {
        for (j = i; j < 3; j++) {
            char_array_3[j] = '\0';
        }

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (j = 0; (j < i + 1); j++) {
            ret += base64_chars[char_array_4[j]];
        }

        while ((i++ < 3)) {
            ret += '=';
        }

    }

    return ret;
}

/***
 * Base64 encode
 * @param input
 * @return
 */
std::string Base64::base64_encode(const std::string& input) {
    std::string output;
    size_t i = 0;
    unsigned char decode_block[3];
    unsigned char encode_block[4];
    for (std::string::size_type len = 0; len != input.size(); ++len) {
        decode_block[i++] = input[len];
        if (i == 3) {
            encode_char_array(encode_block, decode_block);
            for (i = 0; i < 4; ++i) {
                output += base64_chars[encode_block[i]];
            }
            i = 0;
        }
    }
    if (i > 0) {
        for (size_t j = i; j < 3; ++j) {
            decode_block[j] = '\0';
        }
        encode_char_array(encode_block, decode_block);
        for (size_t j = 0; j < i + 1; ++j) {
            output += base64_chars[encode_block[j]];
        }
        while (i++ < 3) {
            output += '=';
        }
    }
    return output;
}

/***
 * Base64 decode
 * @param s
 * @return
 */
std::string Base64::base64_decode(const std::string& encoded_string) {

    size_t in_len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in_ = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::string ret;

    while (in_len-- && ( encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {
        char_array_4[i++] = encoded_string[in_];
        in_++;
        if (i ==4) {
            for (i = 0; i <4; i++) {
                char_array_4[i] = base64_chars.find(char_array_4[i]) & 0xff;
            }

            char_array_3[0] = ( char_array_4[0] << 2       ) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) +   char_array_4[3];

            for (i = 0; (i < 3); i++) {
                ret += char_array_3[i];
            }
            i = 0;
        }
    }

    if (i) {
        for (j = 0; j < i; j++) {
            char_array_4[j] = base64_chars.find(char_array_4[j]) & 0xff;
        }

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);

        for (j = 0; (j < i - 1); j++) {
            ret += char_array_3[j];
        }
    }

    return ret;
}
}
}