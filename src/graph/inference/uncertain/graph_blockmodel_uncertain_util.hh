// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#ifndef GRAPH_BLOCKMODEL_UNCERTAIN_UTIL_HH
#define GRAPH_BLOCKMODEL_UNCERTAIN_UTIL_HH

#include "config.h"

namespace graph_tool
{
using namespace boost;
using namespace std;

struct uentropy_args_t:
        public entropy_args_t
{
    uentropy_args_t(const entropy_args_t& ea)
        : entropy_args_t(ea){}
    bool latent_edges;
    bool density;
};

template <class T>
T logsum(T a, T b)
{
    return std::max(a, b) + log1p(exp(std::min(a, b) - std::max(a, b)));
}

template <class State, class... X>
double get_edge_prob(State& state, size_t u, size_t v, const uentropy_args_t& ea,
                     double epsilon, X... x)
{
    auto e = state.get_u_edge(u, v);
    size_t ew = 0;
    [[maybe_unused]] double old_x = 0;
    if (e != state._null_edge)
    {
        ew = state._eweight[e];
        if constexpr (sizeof...(X) > 0)
            old_x = state._xc[e];
    }

    for (size_t i = 0; i < ew; ++i)
        state.remove_edge(u, v);

    double S = 0;
    double delta = 1. + epsilon;
    size_t ne = 0;
    double L = -std::numeric_limits<double>::infinity();
    while (delta > epsilon || ne < 2)
    {
        double dS = state.add_edge_dS(u, v, x..., ea);
        state.add_edge(u, v, x...);
        S += dS;
        double old_L = L;
        L = logsum(L, -S);
        ne++;
        delta = abs(L-old_L);
    }

    L = (L > 0) ? -log1p(exp(-L)) : L - log1p(exp(L));

    for (int i = 0; i < int(ne - ew); ++i)
        state.remove_edge(u, v);
    for (int i = 0; i < int(ew - ne); ++i)
        if constexpr (sizeof...(X) > 0)
            state.add_edge(u, v, old_x);
        else
            state.add_edge(u, v);

    return L;
}

template <class State>
void get_edges_prob(State& state, python::object edges, python::object probs,
                    const uentropy_args_t& ea, double epsilon)
{
    multi_array_ref<uint64_t,2> es = get_array<uint64_t,2>(edges);
    multi_array_ref<double,1> eprobs = get_array<double,1>(probs);
    for (size_t i = 0; i < eprobs.shape()[0]; ++i)
        eprobs[i] = get_edge_prob(state, es[i][0], es[i][1], ea, epsilon);
}

template <class State>
void get_xedges_prob(State& state, python::object edges, python::object probs,
                    const uentropy_args_t& ea, double epsilon)
{
    multi_array_ref<double,2> es = get_array<double,2>(edges);
    multi_array_ref<double,1> eprobs = get_array<double,1>(probs);
    for (size_t i = 0; i < eprobs.shape()[0]; ++i)
        eprobs[i] = get_edge_prob(state, es[i][0], es[i][1], ea, epsilon,
                                  (es.shape()[1] > 2) ? es[i][2] : 0);
}

template <class State, class Graph, class EProp>
void set_state(State& state, Graph& g, EProp w)
{
    std::vector<std::pair<size_t, size_t>> us;
    for (auto v : vertices_range(state._u))
    {
        us.clear();
        for (auto e : out_edges_range(v, state._u))
        {
            auto w = target(e, state._u);
            if (w == v)
                continue;
            us.emplace_back(w, state._eweight[e]);
        }
        for (auto& uw : us)
        {
            for (size_t i = 0; i < uw.second; ++i)
                state.remove_edge(v, uw.first);
        }
        auto& e = state.template get_u_edge<false>(v, v);
        if (e == state._null_edge)
            continue;
        size_t x = state._eweight[e];
        for (size_t i = 0; i < x; ++i)
            state.remove_edge(v, v);
    }

    for (auto e : edges_range(g))
    {
        for (size_t i = 0; i < size_t(w[e]); ++i)
            state.add_edge(source(e, g), target(e, g));
    }
}


} // graph_tool namespace

#endif //GRAPH_BLOCKMODEL_UNCERTAIN_UTIL_HH
