// author: Mahmoud Khairy, (Purdue Univ)
// email: abdallm@purdue.edu

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef HASHING_H
#define HASHING_H

typedef unsigned long long new_addr_type;

unsigned ipoly_hash_function(new_addr_type higher_bits, unsigned index,
                             unsigned bank_set_num);

unsigned bitwise_hash_function(new_addr_type higher_bits, unsigned index,
                               unsigned bank_set_num);

unsigned PAE_hash_function(new_addr_type higher_bits, unsigned index,
                           unsigned bank_set_num);

unsigned mini_hash_function(new_addr_type higher_bits, unsigned index,
                            unsigned bank_set_num);
#endif