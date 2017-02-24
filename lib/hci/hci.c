/*
 * Slater-Condon rule implementation for Heat-Bath CI
 */

#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include "hci.h"
//#include <omp.h>
#include <limits.h>


void contract_h_c(double *h1, double *eri, int norb, uint64_t *strs, double *civec, double *hdiag, int ndet, double *ci1) {

    size_t ip, jp;
    int nset = norb / 64 + 1;

    printf("Number of orbitals:     %d\n", norb);
    printf("Number of determinants: %d\n", ndet);
    printf("Number of string sets:  %d\n", nset);

    // Loop over pairs of determinants
    for (ip = 0; ip < ndet; ++ip) {
        uint64_t *stria = strs + ip * 2 * nset;
        uint64_t *strib = strs + ip * 2 * nset + nset;
        for (jp = 0; jp < ip; ++jp) {
            uint64_t *strja = strs + jp * 2 * nset;
            uint64_t *strjb = strs + jp * 2 * nset + nset;
            int n_excit_a = n_excitations(stria, strja, nset);
            int n_excit_b = n_excitations(strib, strjb, nset);
            printf("%d %d %d %d %d %d\n", stria[0], strib[0], strja[0], strjb[0], n_excit_a, n_excit_b);
            printf("%s %s %s %s\n", int2bin(stria[0]), int2bin(strib[0]), int2bin(strja[0]), int2bin(strjb[0]));
            // Single excitation
            if ((n_excit_a + n_excit_b) == 1) {
                int *ia;
                // Alpha->Alpha
                if (n_excit_b == 0) {
                    ia = get_single_excitation(stria, strja, nset);
                }
                // Beta->Beta
                else if (n_excit_a == 0) {
                    ia = get_single_excitation(strib, strjb, nset);
                }
                printf("i: %d -> a: %d\n", ia[0], ia[1]);
            }
            // Double excitation
            else if ((n_excit_a + n_excit_b) == 2) {
            }
        }
        if (ip == 8) exit(1);
    }

}


int n_excitations(uint64_t *str1, uint64_t *str2, int nset) {

    size_t p;
    int d = 0;

    for (p = 0; p < nset; ++p) {
        d += popcount(str1[p] ^ str2[p]);
    }

    return d / 2;

}


int popcount(uint64_t x) {

    const uint64_t m1  = 0x5555555555555555; //binary: 0101...
    const uint64_t m2  = 0x3333333333333333; //binary: 00110011..
    const uint64_t m4  = 0x0f0f0f0f0f0f0f0f; //binary:  4 zeros,  4 ones ...
    const uint64_t m8  = 0x00ff00ff00ff00ff; //binary:  8 zeros,  8 ones ...
    const uint64_t m16 = 0x0000ffff0000ffff; //binary: 16 zeros, 16 ones ...
    const uint64_t m32 = 0x00000000ffffffff; //binary: 32 zeros, 32 ones
//    const uint64_t hff = 0xffffffffffffffff; //binary: all ones
//    const uint64_t h01 = 0x0101010101010101; //the sum of 256 to the power of 0,1,2,3...
    x = (x & m1 ) + ((x >>  1) & m1 ); //put count of each  2 bits into those  2 bits 
    x = (x & m2 ) + ((x >>  2) & m2 ); //put count of each  4 bits into those  4 bits 
    x = (x & m4 ) + ((x >>  4) & m4 ); //put count of each  8 bits into those  8 bits 
    x = (x & m8 ) + ((x >>  8) & m8 ); //put count of each 16 bits into those 16 bits 
    x = (x & m16) + ((x >> 16) & m16); //put count of each 32 bits into those 32 bits 
    x = (x & m32) + ((x >> 32) & m32); //put count of each 64 bits into those 64 bits 

    return x;

}


int *get_single_excitation(uint64_t *str1, uint64_t *str2, int nset) {

    size_t p;
    int *ia = malloc(sizeof(int) * 2);

    for (p = 0; p < nset; ++p) {
        uint64_t str_tmp = str1[p] ^ str2[p];
        uint64_t str_particle = str_tmp & str2[p];
        uint64_t str_hole = str_tmp & str1[p];

        if (popcount(str_particle) == 1) {
            ia[1] = trailz(str_particle) + 64 * p;
        }
       
        if (popcount(str_hole) == 1) {
            ia[0] = trailz(str_hole) + 64 * p;
        }

    }

    return ia;

}


int trailz(uint64_t v) {

    int c = 64;

    v &= -(signed) v;
    if (v) c--;
    if (v & 0x00000000ffffffff) c -= 32;
    if (v & 0x0000ffff0000ffff) c -= 16;
    if (v & 0x00ff00ff00ff00ff) c -= 8;
    if (v & 0x0f0f0f0f0f0f0f0f) c -= 4;
    if (v & 0x3333333333333333) c -= 2;
    if (v & 0x5555555555555555) c -= 1;

    return c;
}

// Function to print int as a char for debug purposes
char *int2bin(int i) {
    size_t bits = sizeof(int) * CHAR_BIT;

    char * str = malloc(bits + 1);
    if(!str) return NULL;
    str[bits] = 0;

    // type punning because signed shift is implementation-defined
    unsigned u = *(unsigned *)&i;
    for(; bits--; u >>= 1)
        str[bits] = u & 1 ? '1' : '0';

    return str;
}
