#include <vector>
#include <numeric>
#include "utils.hpp"
#include "bandits.hpp"
#include "policies.hpp"
#include <algorithm>
policy::policy(bandit &bandit_ref) {
    this->bandit_ref = &bandit_ref;
    K = bandit_ref.K;
    D = bandit_ref.D;
    dim = bandit_ref.D;
    sigma = bandit_ref.sigma;
    action_space = bandit_ref.action_space;
}
std::pair<bool, size_t> policy::loop(void) {
    return {};
}
policy_fb::policy_fb(bandit &bandit_ref) {
    this->bandit_ref = &bandit_ref;
    K = bandit_ref.K;
    D = bandit_ref.D;
    dim = bandit_ref.D;
    sigma = bandit_ref.sigma;
    action_space = bandit_ref.action_space;
}
std::pair<std::pair<size_t, bool>, std::vector<size_t>> policy_fb::loop(size_t, size_t) {
    return {};
}

psi_ape::psi_ape(bandit &bandit_ref): policy(bandit_ref) {
};
std::pair<bool, size_t> psi_ape::loop(const size_t& seed, const double& delta) {
    this->delta = delta;
    double cg = Cg(this->delta);
    // Initialize the model
    bandit_ref->reset_env(seed);
    std::vector<size_t> St;
    std::vector<size_t> opt;
    std::vector<bool> opt_mask(K);
    std::vector<bool> St_mask;
    std::vector<size_t> St_comp;
    std::vector<size_t> opt_comp;
    std::vector<size_t> Ts(K, 1);
    std::vector<std::vector<double>> beta (K, std::vector<double>(K, betaij(1, 1, cg, sigma)));
    std::vector<std::vector<double>> means_t(K, std::vector<double>(D));
    double z1_t, z2_t;
// compute the empirical Pareto set St
#define get_St {St_mask  = pareto_optimal_arms_mask(means_t); \
std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(St), [&St_mask]( size_t i){return St_mask[i];}); \
std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(St_comp), [&St_mask]( size_t i){return !St_mask[i];});};
// compute OPT(t)
#define get_opt {std::transform(action_space.begin(), action_space.end(), opt_mask.begin(), [&](size_t i){\
    return get_h(i, means_t, beta, 0.) > 0;\
});for (size_t  i{0}; i < K; ++i) {\
opt_mask[i]?opt.push_back(i): opt_comp.push_back(i);}}

    size_t at, bt, ct;
// Initial sampling
    for (auto k:action_space){
        means_t[k] = bandit_ref->sample({k})[0];
    }
    get_St
    get_opt
    // check stopping rule
    z1_t = get_z1t(means_t, St, beta,0.);
    z2_t = get_z2t(means_t, St_comp, beta, 0.);
    size_t t = K;
    while((z1_t<0 || z2_t <0)){
        bt = get_bt(means_t, opt_comp, beta);
        ct = get_ct(means_t, bt, beta);
//at = (Ts[bt]>Ts[ct])?ct:bt;
        for (auto k: {bt, ct}) {
            std::vector<double> v(std::move(bandit_ref->sample({k})[0])); // to move
            std::transform(means_t[k].begin(), means_t[k].end(), v.begin(), means_t[k].begin(),[&](double mean_t, double xval){
                return (xval + ((double)Ts[k])*mean_t) / ((double)Ts[k] + 1.);
            });
            ++Ts[k];
        }
        // update beta_ij table
        for (auto j:{bt, ct})
        for (size_t i = 0; i < K; ++i) {
            beta[j][i] = betaij(Ts[i], Ts[j],cg, sigma);
            beta[i][j] = beta[j][i];
        }
        St.clear();
        St_comp.clear();
        opt.clear();
        opt_comp.clear();
        get_St
        get_opt
        z1_t = get_z1t(means_t, St, beta,0.);
        z2_t = get_z2t(means_t, St_comp, beta, 0.);
        ++t;
    }
    std::sort(St.begin(), St.end());
    // optimal_arms is always sorted;
    bool is_found = std::equal(St.begin(), St.end(), bandit_ref->optimal_arms.begin(), bandit_ref->optimal_arms.end()); // optimal_set found
return std::pair<bool, size_t>{is_found, std::accumulate(Ts.begin(), Ts.end(), size_t{0})};
};

psi_uniform::psi_uniform(bandit &bandit_ref):policy(bandit_ref) {};

std::pair<bool, size_t> psi_uniform::loop(const size_t& seed, const double& delta)  {
    this->delta = delta;
    // Initialize the model
    bandit_ref->reset_env(seed);
    double cg = Cg(this->delta);
    std::vector<size_t> St;
    std::vector<size_t> opt;
    std::vector<bool> opt_mask(K);
    std::vector<bool> St_mask;
    std::vector<size_t> St_comp;
    std::vector<size_t> opt_comp;
    std::vector<size_t> Ts(K, 1);
    std::vector<std::vector<double>> means_t(K, std::vector<double>(D));
    std::vector<std::vector<double>> beta (K, std::vector<double>(K, betaij(1, 1, cg, sigma)));
    std::mt19937 gen(seed); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<> distrib(0,K-1);
    double z1_t, z2_t;
#define get_St {St_mask  = std::move(pareto_optimal_arms_mask(means_t)); \
std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(St), [&St_mask]( size_t i){return St_mask[i];}); \
std::copy_if(action_space.begin(), action_space.end(), std::back_inserter(St_comp), [&St_mask]( size_t i){return !St_mask[i];});};
#define get_opt {std::transform(action_space.begin(), action_space.end(), opt_mask.begin(), [&](size_t i){\
    return get_h(i, means_t, beta, 0.) > 0;\
});for (size_t  i{0}; i < K; ++i) {\
opt_mask[i]?opt.push_back(i): opt_comp.push_back(i);}}

// Initial sampling
    for (auto k:action_space){
        means_t[k] = bandit_ref->sample({k})[0];
    }
    get_St
    get_opt
    // check stopping rule
    z1_t = get_z1t(means_t, St, beta, 0.);
    z2_t = get_z2t(means_t, St_comp, beta, 0.);
    size_t t = K;
    while(z1_t<0 || z2_t <0){
        size_t kt = distrib(gen);
        for (auto k: action_space) {
            std::vector<double> v(std::move(bandit_ref->sample({k})[0])); // to move
            std::transform(means_t[k].begin(), means_t[k].end(), v.begin(), means_t[k].begin(),[&](double mean_t, double xval){
                return (xval + ((double)Ts[k])*mean_t) / ((double)Ts[k] + 1.);
            });
            ++Ts[k];
        }
        for (size_t i = 0; i < K; ++i) {
            for (size_t j = 0; j < i; ++j) {
                beta[i][j] = betaij(Ts[i], Ts[j], cg, sigma);
                beta[j][i] = beta[i][j];
            }
        }
        St.clear();
        St_comp.clear();
        opt.clear();
        opt_comp.clear();
        get_St
        get_opt
        z1_t = get_z1t(means_t, St, beta, 0.);
        z2_t = get_z2t(means_t, St_comp, beta, 0.);
        ++t;
    }
    std::sort(St.begin(), St.end());
    // optimal_arms is always sorted;
    bool is_found = std::equal(St.begin(), St.end(), bandit_ref->optimal_arms.begin(), bandit_ref->optimal_arms.end()); // optimal_set foundp
    return std::pair<bool, size_t>{is_found, std::accumulate(Ts.begin(), Ts.end(), size_t{0})};
}
psi_auer::psi_auer(bandit &bandit_ref):policy(bandit_ref){};
std::pair<bool, size_t> psi_auer::loop(const size_t& seed, const double& delta) {
    this->delta = delta;
    double cg = Cg(this->delta);
    // Initialize the model
    bandit_ref->reset_env(seed);
    std::vector<size_t> Ts(K, 0);
    std::vector<size_t> A1_t(action_space); // see paper
    std::vector<size_t> P1_t;
    std::vector<size_t> P2_t;
    std::vector<size_t> optimal_arms;
    std::vector<size_t> A1_t_set_minus_P1_t ;
    std::vector<std::vector<double>> mus_t(K, std::vector<double>(D));
    optimal_arms.reserve(K);
    P1_t.reserve(K);
    P2_t.reserve(K);
    A1_t_set_minus_P1_t.reserve(K);
    while(!A1_t.empty()){
        // sample all active arms
        for (auto a: A1_t){
            std::vector<double> v = std::move(bandit_ref->sample({a})[0]);
            std::transform(mus_t[a].begin(), mus_t[a].end(), v.begin(), mus_t[a].begin(),[&](double mean_t, double xval){
                return (xval + (double)(Ts[a])*mean_t) / ((double)Ts[a] + 1.);
            });
            ++Ts[a];
        }
        // update betas
        // update to remove suboptimal arms
        A1_t.erase(std::remove_if(A1_t.begin(), A1_t.end(), [&A1_t, &mus_t, &Ts, this, &cg](size_t i){
            return std::any_of(A1_t.begin(), A1_t.end(), [&Ts, &mus_t, &i, this, &cg](size_t j){
                return (std::max(minimum_quantity_non_dom(mus_t[i], mus_t[j], 0.),0.)  > (betaij(Ts[i], Ts[j], cg,sigma)));
            });
        }), A1_t.end());
        // compute P1_t
        std::copy_if(A1_t.begin(), A1_t.end(), std::back_inserter(P1_t), [&A1_t, &mus_t, &Ts, this, &cg](size_t i){
            return std::all_of(A1_t.begin(), A1_t.end(), [&](size_t j){
                return (std::max(minimum_quantity_dom(mus_t[i], mus_t[j], 0.),0.) + INF*(i==j)) >(betaij(Ts[i], Ts[j], cg, sigma));
            });
        });
        // recover the set A_1 \P_1
        std::copy_if(A1_t.begin(), A1_t.end(), std::back_inserter(A1_t_set_minus_P1_t), [&P1_t](size_t i){
            return (std::find(P1_t.begin(), P1_t.end(), i) == P1_t.end());
        });
        // compute P2 (see paper)
        if (A1_t_set_minus_P1_t.empty()){
            P2_t = P1_t;
        }
        else {
            std::copy_if(P1_t.begin(), P1_t.end(), std::back_inserter(P2_t), [&mus_t, &Ts, &A1_t_set_minus_P1_t, this, &cg](size_t i){
                return !any_of(A1_t_set_minus_P1_t.begin(), A1_t_set_minus_P1_t.end(), [&, this](size_t j ){
                    return std::max(minimum_quantity_dom(mus_t[j], mus_t[i], 0.),0.) <=(betaij(Ts[i], Ts[j], cg, sigma));
                });
            });
        }
        if (!P2_t.empty()){
            A1_t.erase(std::remove_if(A1_t.begin(), A1_t.end(), [&P2_t](size_t i){
                return (std::find(P2_t.begin(), P2_t.end(), i)!= P2_t.end());
            }), A1_t.end());
            std::copy(P2_t.begin(), P2_t.end(), std::back_inserter(optimal_arms));}
        // clear all the data
        P1_t.clear();
        P2_t.clear();
        A1_t_set_minus_P1_t.clear();
    }
    std::sort(optimal_arms.begin(), optimal_arms.end());
    // pareto_optimal_arms is always sorted;
    // checking correctness if further done in Python
    bool is_found = std::equal(optimal_arms.begin(), optimal_arms.end(), bandit_ref->optimal_arms.begin(), bandit_ref->optimal_arms.end()); // optimal_set found
    return std::pair<bool, size_t>{is_found, std::accumulate(Ts.begin(), Ts.end(), size_t{0})};
}


ege_sr::ege_sr(bandit &bandit_ref):policy_fb(bandit_ref){};


bool ege_sr::loop(const size_t seed, const size_t T) {
    bandit_ref->reset_env(seed);
    std::vector<size_t> n_ks(K);
    std::vector<bool> active_mask(K, true);
    std::vector<size_t> Nc(K, 0);
    std::vector<bool> accept_mask(K, false);
    std::vector<std::vector<double>> total_rewards(K, std::vector<double>(D, 0.));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    std::vector<bool> Sr_mask(K);
    // std::vector<size_t> to_pull;
    std::vector<std::vector<double>> batch_pulls;
    std::vector<double> vec_sub_gap(K);
    std::vector<double> vec_delta_star(K);
    size_t opt_arms_found{0};
    size_t i_r;
    double log_K = std::accumulate(action_space.begin(), action_space.end(), -1./2, []( double acc, size_t i ){return acc + 1./(double)(i+1); });
    double I_1, I_2;
    size_t a_r, d_r;
    size_t num_pulls ;
    n_ks[0] = 0;
    for (size_t r = 1; r < K; ++r) {
        n_ks[r] = std::ceil((1./log_K)* (T-K) /(K+1. - r));
    }
    for (size_t r{1}; r < K; ++r) {
        num_pulls = n_ks[r] - n_ks[r-1];
        if (num_pulls>0){
            for (size_t i {0}; i < K; ++i) {
                if (active_mask[i]){
                    // pull active arms
                    batch_pulls = bandit_ref->sample(std::vector<size_t>(num_pulls, i));
                    for(size_t t{0}; t<num_pulls; ++t){
                        for(size_t d{0}; d<D; ++d) total_rewards[i][d] += batch_pulls[t][d];
                    }
                    Nc[i] += num_pulls;
                    // actualize the empirical means
                    for (size_t d{0}; d < D; ++d) {
                        means[i][d] = total_rewards[i][d] / (double) Nc[i];
                    }
                }
            }
        }
        Sr_mask = pareto_optimal_arms_mask(means, active_mask);
        vec_delta_star = std::move(delta_star(means, active_mask));
        I_1 = -INF;
        I_2 = -INF;
        for (size_t a:action_space) {
            if (active_mask[a]){
                vec_sub_gap[a] = sub_opt_gap(a, means, vec_delta_star, active_mask, Sr_mask[a]);
                switch (Sr_mask[a]) {
                    case true:
                        // optimal
                        if (vec_sub_gap[a]>I_1) {
                            a_r = a; I_1 = vec_sub_gap[a];
                        }
                        break;
                    case false: // sub-optimal
                        if (vec_sub_gap[a]>I_2) {
                            d_r = a; I_2 = vec_sub_gap[a];
                        }
                        break;
                }
            }
        }
        if (I_2>=I_1){
            // reject sub-optimal arm
            i_r = d_r;
            accept_mask[i_r] = false;
        }
        else {
            i_r = a_r;
            accept_mask[i_r] = true;
        }
        // deactivate the removed arm
        active_mask[i_r] = false;
    }
    accept_mask = std::move(sum(accept_mask, active_mask));
    for(auto a:action_space){
        opt_arms_found += (accept_mask[a] && accept_mask[a]==bandit_ref->optimal_arms_mask[a]);
    }
    return  std::equal(accept_mask.begin(), accept_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
}


ege_sh::ege_sh(bandit &bandit_ref):policy_fb(bandit_ref){};

bool ege_sh::loop(const size_t seed, const size_t T) {
    bandit_ref->reset_env(seed);
    std::vector<bool> active_mask(K, true);
    std::vector<size_t> Nc(K, 0);
    std::vector<bool> accept_mask(K, false);
    std::vector<std::vector<double>> total_rewards(K, std::vector<double>(D, 0.));
    std::vector<std::vector<double>> means(K, std::vector<double>(D));
    std::vector<size_t> temp_vec{action_space};
    std::vector<bool> Sr_mask(K);
    size_t ceil_log_K  = std::ceil(std::log2(K));
    size_t Nr, Nr_keep, Nr_remove; // number of active arms, number of arms to keep and to remove
    std::vector<std::vector<double>> batch_pulls;
    std::vector<double> vec_sub_gap(K);
    std::vector<double> vec_delta_star(K);
    double threshold;
    size_t count_remove, counter, a_, num_pulls;
    for (size_t r{0}; r < ceil_log_K; ++r) {
        Nr = std::accumulate(active_mask.begin(), active_mask.end(), size_t{0});
        num_pulls = std::floor(T/((double)(Nr*ceil_log_K)));
        if (num_pulls>0){
            for (size_t i{0}; i < K; ++i) {
                if (active_mask[i]){
                    // pull active arms
                    batch_pulls = bandit_ref->sample(std::vector<size_t>(num_pulls, i));
                    for(size_t t{0}; t<num_pulls; ++t){
                        for(size_t d{0}; d<D; ++d) total_rewards[i][d] += batch_pulls[t][d];
                    }
                    Nc[i] += num_pulls;
                    for (size_t d{0}; d < D; ++d) {
                        means[i][d] = total_rewards[i][d] / (double) Nc[i];
                    }
                }
            }
        }
        Sr_mask = pareto_optimal_arms_mask(means, active_mask);
        vec_delta_star = delta_star(means, active_mask);
        for(size_t a: action_space){
            if (active_mask[a]){
                vec_sub_gap[a] = sub_opt_gap(a, means, vec_delta_star, active_mask, Sr_mask[a]) ;//+1e-4*(!Sr_mask[a]);
            }
        }
        Nr_keep = std::ceil((double)Nr/ 2.);
        Nr_remove = Nr - Nr_keep;
        // tie-breaking rule
        std::sort(temp_vec.begin(), temp_vec.end(), [&vec_sub_gap, &active_mask](size_t i, size_t j){
            return vec_sub_gap[i] - INF*(!active_mask[i]) < vec_sub_gap[j] - INF*(!active_mask[j]);
        });
        // any arm with a gap larger than this threshold will be discarded
        threshold = vec_sub_gap[temp_vec[(K-Nr) + Nr_keep]];
        try
        {
            std::sort(temp_vec.begin()+(K-Nr), temp_vec.end(),  [&Sr_mask, &vec_sub_gap](size_t i, size_t j){
                double x{vec_sub_gap[i] - (!Sr_mask[i])*INF}, y{vec_sub_gap[j] -(!Sr_mask[j])*INF};
                if (x==y) return i<j;
                return  x<y;});
        }
        catch (std::exception& e)
        {
            std::cout << "Standard exception: " << e.what() << std::endl;
        }
        count_remove = 0;
        counter = 0;

        while(count_remove< Nr_remove){
            a_ = temp_vec[counter];
           if( active_mask[a_] && vec_sub_gap[a_]>= threshold){
               accept_mask[a_] = Sr_mask[a_];
               active_mask[a_] = false;
               ++count_remove;
           }
           ++counter;
        }

    }
    accept_mask = std::move(sum(accept_mask, active_mask));
    return std::equal(accept_mask.begin(), accept_mask.end(), bandit_ref->optimal_arms_mask.begin(), bandit_ref->optimal_arms_mask.end()); // optimal_set found
}




std::vector<std::vector<bool>> batch_sr(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds){
    std::vector<std::vector<bool>> ans(Ts.size(), std::vector<bool>(seeds.size()));
    //thread_local
    // size_t N{seeds.size()};
    ege_sr sr(bandit_ref);
    size_t count{0};
    size_t i, j;
    for(i=0;i<Ts.size(); ++i){
        count = 0;
#pragma omp parallel for default(none)  shared(Ts, seeds, i, k) firstprivate(sr)  reduction(+:count)
        for(j=0; j<seeds.size(); ++j){
            ans[i][j] = (size_t)sr.loop(seeds[j], Ts[i]);
        }
//#pragma omp barrier
    }
    return ans;
}

std::vector<std::vector<bool>> batch_sh(bandit& bandit_ref, std::vector<size_t>& Ts, std::vector<size_t>& seeds){
    size_t count;
    std::vector<std::vector<bool>> ans(Ts.size(), std::vector<bool>(seeds.size()));
    //thread_local
    size_t N{seeds.size()};
    ege_sh sh(bandit_ref);
    size_t i;
    size_t j;
    for(i=0;i<Ts.size(); ++i){
        count = 0;
#pragma omp parallel for private(j) firstprivate(sh) default(none) shared(seeds, Ts, i) reduction(+:count)
        for(j=0; j<seeds.size(); ++j){
            ans[i][j]=  sh.loop(seeds[j], Ts[i]);
        }
        //ans[i] = (double) count / (double) N;
    }
    return ans;
}

std::vector<std::pair<bool,size_t>> batch_ape(bandit& bandit_ref, double delta, std::vector<size_t>& seeds){
    size_t count;
    std::vector<std::pair<bool, size_t>> ans(seeds.size());
    //thread_local
    size_t N{seeds.size()};
    psi_ape ape(bandit_ref);
    size_t j;
#pragma omp parallel for private(j) firstprivate(ape) default(none) shared(seeds) reduction(+:count)
        for(j=0; j<seeds.size(); ++j){
            ans[j]=  ape.loop(seeds[j], delta);
        }
        //ans[i] = (double) count / (double) N;
    return ans;
}
