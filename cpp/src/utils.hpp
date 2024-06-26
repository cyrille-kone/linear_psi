#pragma once
#include<vector>
#include<cstddef>
#include <cstdlib>
#include <numeric>
#include <cmath>
#include <cassert>
#include<iostream>
#include <algorithm>
#define INF 1e12
#define EE 2.71828182845904523536
#define get_argmin(v, idx) (idx)[std::distance((v).begin(), std::min_element((v).begin(), (v).end()))]
#define get_argmax(v, idx) (idx)[std::distance((v).begin(), std::max_element((v).begin(), (v).end()))]
#define in_set(id, v) (std::find((v).begin(), (v).end(), (id)) != (v).end())

// utility functions for arrays
std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>>&means);
std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>>&means_t, const std::vector<double>& betas, double eps);
inline double minimum_quantity_dom(const std::vector<double>& xi, const std::vector<double>& xj, const double eps){
    double res{-INF};
    for (size_t k{0}; k < xi.size(); ++k)
        res = std::max(res, xi[k] + eps - xj[k]);
    return res;
}
inline double minimum_quantity_non_dom(const std::vector<double> & xi, const std::vector<double>& xj, const double eps){
    double res{INF};
    for (size_t k{0}; k < xi.size(); ++k)
        res = std::min(res, xj[k] - xi[k] + eps );
    return res;
}
/*
 * Return true if xi is dominated by xj
 */
inline bool is_pareto_dominated(const std::vector<double>& xi, const std::vector<double>& xj, const double& eps){
    bool is_strict{false};
    for (size_t k{0}; k<xi.size(); ++k){
        if (xi[k] + eps > xj[k]) return false;
        is_strict |= (xi[k] + eps < xj[k]);
    }
    return is_strict;
};

// @reference Kaufmann et al. 2021
inline double Cg(const double& delta){
    return log(1/delta)/2 + std::max(log(log(1/delta)/2), 0.);
}
// taken from Kaufmann et. al 2021 simplified for empirical purpose
inline double betaij(const size_t& Ti, const size_t& Tj, const double& cg, const double& sigma){
    return sigma*sqrt(
            2*( 2*cg + 1.*log(1.+log(Ti)) + 1.*log(1.+log(Tj))
            )*(
                    (1./Ti) + (1./Tj)
            )
    );
}
// @reference Katz-Samuels et al. 2019
inline double beta_lil(const size_t& Ti, const size_t& Tj, const double& cg, const double& sigma){
    return sigma*(sqrt(( cg + 3.*log(log(EE*(double)Ti))) /  (double)Ti) + sqrt(( cg + 3.*log(log(EE*(double)Tj))) /  (double)Tj));
}
[[nodiscard]] inline double get_g(const size_t& i, const std::vector<std::vector<double>>& means, const std::vector<std::vector<double>>& beta, const std::vector<size_t>& opt, const double& eps_2){
    double res{-INF};
    size_t K{means.size()};
    for(size_t j{0}; j<K; ++j)
        res = std::max(res, minimum_quantity_non_dom(means[i], means[j],0.) - beta[i][j] + eps_2*((eps_2>0) && in_set(j, opt)) -INF*(i==j));
    return res;
}
[[nodiscard]] inline double get_h(const size_t& i, const std::vector<std::vector<double>>& means, const std::vector<std::vector<double>>& beta, const double& eps_1){
    double res{INF};
    size_t K{means.size()};
    for(size_t j{0}; j<K; ++j)
        res = std::min(res, minimum_quantity_dom(means[i], means[j], eps_1) - beta[i][j] + INF*(i==j));
    return  res;
}

// @utility function
inline double M(const std::vector<double>& xi, const std::vector<double>& xj) {
    double res{-INF};
    for (size_t k{0}; k < xi.size(); ++k)
        res = std::max(res, xi[k]  - xj[k]);
    return res;
}
// @utility function
inline double m(const std::vector<double>& xi, const std::vector<double>& xj){
    double res{INF};
    for (size_t k{0}; k < xi.size(); ++k)
        res = std::min(res, xj[k] - xi[k]);
    return res;
}

std::vector<bool> pareto_optimal_arms_mask(const std::vector<std::vector<double>> &means, const std::vector<bool>& active_mask);
double get_z1t(const std::vector<std::vector<double>> &means, const std::vector<size_t>& St, const std::vector<std::vector<double>>& beta, const double& eps_1);
double get_z2t(const std::vector<std::vector<double>> & means, const std::vector<size_t> & St_comp, const std::vector<std::vector<double>>& beta, const double& eps_1);
std::vector<double> compute_gap(const std::vector<bool>& pareto_set_mask, const std::vector<std::vector<double>>&arms_means);

//@utility function
double sub_opt_gap(size_t i, const std::vector<std::vector<double>>& means, const std::vector<double>& vec_delta_star, const std::vector<bool>& active_mask, bool par);
// @utility function
std::vector<double> delta_star(const std::vector<std::vector<double>>& means, const std::vector<bool>& active_mask);
//@utility function
template <typename T>
std::vector<T> matmul(const std::vector<std::vector<T>>& A, const std::vector<T>& X){
    std::vector<T> res(A.size(), 0);
    assert(A[0].size() == X.size());
    for (size_t i = 0; i < A.size(); ++i) {
        for (int j = 0; j < A[0].size(); j++) {
            res[i] += A[i][j]*X[j];
        }
    }
    return res;
}
template <typename T>
inline std::vector<T> sum(const std::vector<T> x1, const std::vector<T> x2){
    assert(x1.size()== x2.size());
    std::vector<T> res(x1.size());
    for (size_t i =0; i<x1.size(); ++i){
        res[i] = x1[i]+  x2[i];
    }
    return res;
}
template <typename T>
inline std::vector<T> prod(const std::vector<T> x1, const std::vector<T> x2){
    assert(x1.size()== x2.size());
    std::vector<T> res(x1.size());
    for (size_t i =0; i<x1.size(); ++i){
        res[i] = x1[i]*  x2[i];
    }
    return res;
}
template <typename T>
inline std::vector<double> divide(const std::vector<T> x1, const std::vector<T> x2){
    assert(x1.size()== x2.size());
    std::vector<T> res(x1.size());
    for (size_t i =0; i<x1.size(); ++i){
        res[i] = (double)x1[i]/x2[i];
    }
    return res;
}

// debugging
template <typename T>
void print_array1d(std::vector<T>& vec){
    std::cout<<"[ ";
    for (auto t: vec)  std::cout<<t<<", ";
    std::cout<<"]"<<std::endl;
}
template<typename  T>
void print_array2d(std::vector<std::vector<T>>& vec){
    std::cout<<"[ ";
    for (auto t: vec) print_array1d(t);
    std::cout<<"]"<<std::endl;
}
template <typename T>
void print_array1d(const std::vector<T>& vec){
    std::cout<<"[ ";
    for (auto t: vec)  std::cout<<t<<", ";
    std::cout<<"]"<<std::endl;
}
template<typename  T>
void print_array2d(const std::vector<std::vector<T>>& vec){
    std::cout<<"[ ";
    for (auto t: vec) print_array1d(t);
    std::cout<<"]"<<std::endl;
}
