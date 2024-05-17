#pragma once
#include<cstddef>
#include<iostream>
#include "utils.hpp"
#include "bandits.hpp"
struct policy{
    size_t K;
    size_t dim;
    size_t D;
    double sigma;
    std::vector<size_t> action_space;
    bandit* bandit_ref;
    policy() = default;
    explicit policy( bandit&);
    std::pair<bool, size_t> loop();
};
struct psi_auer: policy{
    double delta;
    psi_auer()=default;
    explicit psi_auer(bandit&);
    [[nodiscard]] std::pair<bool, size_t> loop(const size_t&, const double&);
};

struct psi_ape: policy{
    double delta;
    psi_ape()= default;
    explicit psi_ape(bandit&);
    [[nodiscard]] std::pair<bool, size_t> loop(const size_t&, const double&);
    size_t get_ct(const std::vector<std::vector<double>> & means, std::size_t bt, const std::vector<std::vector<double>>& beta) {
        std::vector<double> tmp(K);
        std::transform(action_space.begin(), action_space.end(), tmp.begin(),[&](size_t i){
            return minimum_quantity_dom(means[bt], means[i], 0) - beta[bt][i] + INF*(i==bt);
        });
        return get_argmin(tmp, action_space);}
    size_t get_bt(const std::vector<std::vector<double>>& means, const std::vector<size_t>& opt_comp, const std::vector<std::vector<double>>& beta){
        std::vector<double> res;
        res.reserve(opt_comp.size());
        double res_;
        std::transform(opt_comp.begin(), opt_comp.end(), std::back_inserter(res), [&](size_t i){
            res_ = INF;
            for(size_t j:action_space){
                res_ = std::min(res_, minimum_quantity_dom(means[i], means[j], 0.) + beta[j][i] + INF*(i==j));
            }
            return res_;
        });
        return (opt_comp)[std::distance(res.begin(), std::max_element(res.begin(), res.end()))];
        //return res;

    }
};
struct psi_uniform: policy{
    double delta;
    psi_uniform()= default;
    explicit psi_uniform(bandit&);
    [[nodiscard]] std::pair<bool, size_t> loop(const size_t&, const double&);
};

struct policy_fb{
    size_t K;
    size_t dim;
    size_t D;
    double sigma;
    std::vector<size_t> action_space;
    bandit* bandit_ref;
    policy_fb() = default;
    explicit policy_fb( bandit&);
    std::pair<std::pair<size_t, bool>, std::vector<size_t>> loop(const size_t seed, const size_t T);
};

struct ege_sh: policy_fb {
    ege_sh() = default;
    explicit ege_sh(bandit&);
    [[nodiscard]] bool loop(const size_t seed, const size_t T);
};

struct ege_sr: policy_fb {
    ege_sr() = default;
    explicit ege_sr(bandit&);
    [[nodiscard]] bool loop(const size_t seed, const size_t T);
};
std::vector<std::pair<bool,size_t>> batch_ape(bandit& bandit_ref, double delta, std::vector<size_t>& seeds);
std::vector<std::vector<bool>> batch_sr(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds);
std::vector<std::vector<bool>> batch_sh(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds);