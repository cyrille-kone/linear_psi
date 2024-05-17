#include <vector>
#include "cpp/src/utils.hpp"
#include "cpp/src/bandits.hpp"
#include "cpp/src/policies.hpp"
#include <iostream>
#include <cstdlib>

int main(int argc, char** argv) {
    std::vector<std::vector<double>> data{{0.75785738, 0.04866446},
    {0.4445091 , 0.19905503},
    {0.45422223, 0.9413393 },
    {0.89876041, 0.05671574},
    {0.80762673, 0.22822175},};
    size_t  K = 5, D=2;
    size_t i, k{0};
    //gaussian gauss(data, arr);
    bernoulli bern(data);
    psi_auer auer(bern);
    //psi_ape ape(gauss);
    //psi_uniform psi_unif(gauss);
    ege_sr sr(bern);
    std::cout<<bern.H<<std::endl;
    size_t result{0};
    /*
    for(i=k;i<500; ++i){
        //cout<<auer.loop(i, 0.1, 0).first<<endl;
        auto res = ape.loop(i, 0.1);
        std::cout<<"[seed="<<i<<"]: "<<res.first<<" "<<std::boolalpha<<res.second<<std::endl;
        //auto ret_val = bandit.sample({0,1,2});
        //print_array2d(ret_val);
        result += res.second;
    }
    std::cout<<result/500<<std::endl;*/
    for(i=0;i<500; ++i){
        auto res = sr.loop(i, 500);
        std::cout<<res<<std::endl;
        result += res;
    }
    std::cout<<result/500.<<std::endl;
    std::cout<<bern.H<<std::endl;

    return 0;
}