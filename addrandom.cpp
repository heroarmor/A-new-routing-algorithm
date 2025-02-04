#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <cmath>
#include <bitset>
#include <functional>
#include <cmath>
using namespace std;

// Constants
static const int MAXN = 1600;  // Assume maximum number of nodes does not exceed 1600

using Graph = vector<vector<int>>;

// Function to generate minimal grid network
Graph generate_minimal_grid_network(int rows, int cols) {
    int nodes_per_tile = 10;
    int total_nodes = rows * cols * nodes_per_tile;
    Graph graph(total_nodes, vector<int>(total_nodes, 0));

    auto node_id = [&](int r, int c, int sub) {
        return (r * cols + c) * nodes_per_tile + sub;
    };

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int local_in  = node_id(row, col, 0);
            int local_out = node_id(row, col, 1);

            int west_in   = node_id(row, col, 2);
            int west_out  = node_id(row, col, 3);
            int south_in  = node_id(row, col, 4);
            int south_out = node_id(row, col, 5);
            int east_in   = node_id(row, col, 6);
            int east_out  = node_id(row, col, 7);
            int north_in  = node_id(row, col, 8);
            int north_out = node_id(row, col, 9);

            // Initial internal connections (minimum required edges)
            graph[local_in][west_out]  = 1;
            graph[local_in][south_out] = 1;
            graph[local_in][east_out]  = 1;
            graph[local_in][north_out] = 1;

            graph[west_in][local_out]  = 1;
            graph[south_in][local_out] = 1;
            graph[east_in][local_out]  = 1;
            graph[north_in][local_out] = 1;

            // Connections to adjacent tiles
            if (col > 0) {
                int west_neighbor_east_in = node_id(row, col - 1, 6);
                graph[west_out][west_neighbor_east_in] = 1;
            }
            if (row < rows - 1) {
                int south_neighbor_north_in = node_id(row + 1, col, 8);
                graph[south_out][south_neighbor_north_in] = 1;
            }
            if (col < cols - 1) {
                int east_neighbor_west_in = node_id(row, col + 1, 2);
                graph[east_out][east_neighbor_west_in] = 1;
            }
            if (row > 0) {
                int north_neighbor_south_in = node_id(row - 1, col, 4);
                graph[north_out][north_neighbor_south_in] = 1;
            }
        }
    }
    return graph;
}

// Function to count edges in the graph
int count_edges(const Graph &g) {
    int c = 0;
    for (const auto &row : g) {
        for (int x : row) {
            c += x;
        }
    }
    return c;
}

// Function to compute initial reachability using transitive closure
void compute_initial_paths(const Graph &g, vector<bitset<MAXN>> &Paths) {
    int n = g.size();
    for (int i = 0; i < n; i++) {
        Paths[i].reset();
    }
    // Set direct edges
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (g[i][j] == 1) {
                Paths[i].set(j);
            }
        }
    }
    // Compute transitive closure using Floyd-Warshall
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            if (Paths[i].test(k)) {
                Paths[i] |= Paths[k];
            }
        }
    }
}

// Function to check if the graph is acyclic using Floyd-Warshall
bool floyd_warshall_is_acyclic(const Graph &g) {
    int n = g.size();
    vector<vector<int>> reachability(n, vector<int>(n, 0));

    // Initialize the reachability matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            reachability[i][j] = g[i][j];  // Direct connection edges
        }
    }

    // Compute transitive closure using Floyd-Warshall algorithm
    for (int k = 0; k < n; ++k) {
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                reachability[i][j] = reachability[i][j] || (reachability[i][k] && reachability[k][j]);
            }
        }
    }

    // Check if there exists a path from i to i (cycle detection)
    bool has_cycle = false;
    for (int i = 0; i < n; ++i) {
        if (reachability[i][i] == 1) {  // A node can reach itself, indicating a cycle
            has_cycle = true;
            break;  // No need to continue checking once a cycle is found
        }
    }

    if (has_cycle) {
        cout << "Cycle detected" << endl;
        return false;
    } else {
        cout << "No cycle detected" << endl;
        return true;
    }
}

// Function to check constraints (e.g., out-degree <= 4)
bool check_constraints(const Graph &g) {
    int n = g.size();
    for (int i = 0; i < n; ++i) {
        int out_degree = 0;
        for (int j = 0; j < n; ++j) {
            out_degree += g[i][j];
        }
        if (out_degree > 4) return false; // Example constraint: out-degree cannot exceed 4
    }
    return true;
}

// Function to check if adding edge (u->v) would create a cycle
bool would_create_cycle(int u, int v, const vector<bitset<MAXN>> &Paths) {
    // If Paths[v][u] is true, then adding (u->v) would create a cycle
    return Paths[v].test(u);
}

// Function to add edge and update reachability incrementally
void add_edge_and_update_paths(int u, int v,
                               Graph &g,
                               vector<bitset<MAXN>> &Paths)
{
    g[u][v] = 1; // Add edge to the graph
    int n = g.size();

    // Find all nodes that can reach u
    bitset<MAXN> fromSet;
    fromSet.reset();
    for (int x = 0; x < n; x++) {
        if (Paths[x].test(u) || x == u) {
            fromSet.set(x);
        }
    }

    // Find all nodes that can be reached from v
    bitset<MAXN> toSet;
    toSet.reset();
    for (int y = 0; y < n; y++) {
        if (Paths[v].test(y) || y == v) {
            toSet.set(y);
        }
    }

    // Update reachability: for all x in fromSet, Paths[x] |= toSet
    for (int x = 0; x < n; x++) {
        if (fromSet.test(x)) {
            Paths[x] |= toSet;
        }
    }
}

// Function to collect candidate internal edges
vector<pair<int,int>> collect_candidate_internal_edges(int rows, int cols) {
    vector<pair<int,int>> candidates;
    auto node_id = [&](int r, int c, int sub) {
        return (r * cols + c) * 10 + sub;
    };
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int west_in   = node_id(r, c, 2);
            int west_out  = node_id(r, c, 3);
            int south_in  = node_id(r, c, 4);
            int south_out = node_id(r, c, 5);
            int east_in   = node_id(r, c, 6);
            int east_out  = node_id(r, c, 7);
            int north_in  = node_id(r, c, 8);
            int north_out = node_id(r, c, 9);

            // Possible internal edges as per description
            candidates.emplace_back(west_in, north_out);
            candidates.emplace_back(south_in, north_out);
            candidates.emplace_back(east_in, north_out);

            candidates.emplace_back(north_in, south_out);
            candidates.emplace_back(east_in, south_out);
            candidates.emplace_back(west_in, south_out);

            candidates.emplace_back(west_in, east_out);
            candidates.emplace_back(north_in, east_out);
            candidates.emplace_back(south_in, east_out);

            candidates.emplace_back(east_in, west_out);
            candidates.emplace_back(north_in, west_out);
            candidates.emplace_back(south_in, west_out);
        }
    }
    return candidates;
}

// Helper function to map node index to (row, col, node_type)
tuple<int, int, string> get_node_info(int node, int rows, int cols) {
    int nodes_per_tile = 10;
    int tile_id = node / nodes_per_tile;
    int sub = node % nodes_per_tile;
    int r = tile_id / cols;
    int c = tile_id % cols;
    string type;
    switch (sub) {
        case 0: type = "local_in"; break;
        case 1: type = "local_out"; break;
        case 2: type = "west_in"; break;
        case 3: type = "west_out"; break;
        case 4: type = "south_in"; break;
        case 5: type = "south_out"; break;
        case 6: type = "east_in"; break;
        case 7: type = "east_out"; break;
        case 8: type = "north_in"; break;
        case 9: type = "north_out"; break;
        default: type = "unknown"; break;
    }
    return make_tuple(r, c, type);
}

// Function to reconstruct all shortest paths using DFS
void reconstruct_paths(int current, int source, const vector<vector<int>> &predecessors, vector<int> &path, vector<vector<int>> &all_paths) {
    if (current == source) {
        vector<int> complete_path = path;
        complete_path.push_back(current);
        all_paths.emplace_back(complete_path);
        return;
    }
    for (auto pred : predecessors[current]) {
        path.push_back(pred);
        reconstruct_paths(pred, source, predecessors, path, all_paths);
        path.pop_back();
    }
}

// Function to perform reachability check with BFS and output paths,
// and count the total number of shortest paths.
bool reachibility_check2(const Graph &g, int rows, int cols, double &best) {
    auto node_id = [&](int r, int c, int sub) -> int {
        return (r * cols + c) * 10 + sub;
    };
    bool all_reachable = true;
    int n = g.size();
    best = 0; // Count the total number of shortest paths

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int local_in = node_id(r, c, 0);
            // Iterate over all other tiles
            for (int r2 = 0; r2 < rows; r2++) {
                for (int c2 = 0; c2 < cols; c2++) {
                    if (r2 == r && c2 == c) continue; // Exclude self
                    int local_out = node_id(r2, c2, 1);

                    // BFS from local_in to local_out
                    vector<int> distance(n, -1);
                    vector<vector<int>> predecessors(n, vector<int>());
                    queue<int> q;

                    distance[local_in] = 0;
                    q.push(local_in);

                    while (!q.empty()) {
                        int u = q.front(); q.pop();
                        for (int v = 0; v < n; v++) {
                            if (g[u][v]) {
                                if (distance[v] == -1) {
                                    distance[v] = distance[u] + 1;
                                    predecessors[v].push_back(u);
                                    q.push(v);
                                }
                                else if (distance[v] == distance[u] + 1) {
                                    predecessors[v].push_back(u);
                                }
                            }
                        }
                    }

                    if (distance[local_out] == -1) {
                        all_reachable = false;
                        // Uncomment the next line to output unreachable information:
                        // cout << "local_in(" << r << "," << c << ") cannot reach local_out(" << r2 << "," << c2 << ")\n";
                    }
                    else {
                        // Reconstruct all shortest paths
                        vector<vector<int>> all_paths;
                        vector<int> path;
                        reconstruct_paths(local_out, local_in, predecessors, path, all_paths);

                        // Count the total number of shortest paths
                        best += all_paths.size();

                        // Output information for this pair
                        cout << "local_in(" << r << "," << c << ") -> local_out(" << r2 << "," << c2 << "):";
                        cout << " Shortest path length = " << distance[local_out];
                        cout << ", Number of shortest paths = " << all_paths.size() << endl;
                    }
                }
            }
        }
    }

    return all_reachable;
}

// Function to perform reachability check with BFS and accumulate a weighted sum,
// where the weight is log(number of shortest paths). This represents the sum of
// (log(number of shortest paths)) over all node pairs.
bool reachibility_check(const Graph &g, int rows, int cols, double &total_weighted_paths) {
    auto node_id = [&](int r, int c, int sub) -> int {
        return (r * cols + c) * 10 + sub;
    };
    bool all_reachable = true;
    int n = g.size();
    total_weighted_paths = 0.0; // Sum of (log(number of shortest paths)) for each pair

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int local_in = node_id(r, c, 0);
            for (int r2 = 0; r2 < rows; r2++) {
                for (int c2 = 0; c2 < cols; c2++) {
                    if (r2 == r && c2 == c) continue; // Exclude self
                    int local_out = node_id(r2, c2, 1);

                    // BFS from local_in to local_out
                    vector<int> distance(n, -1);
                    vector<vector<int>> predecessors(n, vector<int>());
                    queue<int> q;

                    distance[local_in] = 0;
                    q.push(local_in);

                    while (!q.empty()) {
                        int u = q.front(); q.pop();
                        for (int v = 0; v < n; v++) {
                            if (g[u][v]) {
                                if (distance[v] == -1) {
                                    distance[v] = distance[u] + 1;
                                    predecessors[v].push_back(u);
                                    q.push(v);
                                }
                                else if (distance[v] == distance[u] + 1) {
                                    predecessors[v].push_back(u);
                                }
                            }
                        }
                    }

                    if (distance[local_out] == -1) {
                        all_reachable = false;
                        // Uncomment the next line to output unreachable information:
                        // cout << "local_in(" << r << "," << c << ") cannot reach local_out(" << r2 << "," << c2 << ")\n";
                    }
                    else {
                        // Reconstruct all shortest paths
                        vector<vector<int>> all_paths;
                        vector<int> path;
                        reconstruct_paths(local_out, local_in, predecessors, path, all_paths);

                        // Accumulate the weighted sum: using log(number of shortest paths)
                        total_weighted_paths += log(all_paths.size());
                    }
                }
            }
        }
    }

    return all_reachable;
}

// A simple reachability check that only verifies if each local_in can reach every local_out
bool reachibility_check1(const Graph &g, int rows, int cols) {
    auto node_id = [&](int r, int c, int sub) -> int {
        return (r * cols + c) * 10 + sub;
    };
    bool all_reachable = true;
    int n = g.size();

    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            int local_in = node_id(r, c, 0);
            // Iterate over all other tiles
            for (int r2 = 0; r2 < rows; r2++) {
                for (int c2 = 0; c2 < cols; c2++) {
                    if (r2 == r && c2 == c) continue; // Exclude self
                    int local_out = node_id(r2, c2, 1);

                    // BFS from local_in to local_out
                    vector<int> distance(n, -1);
                    vector<vector<int>> predecessors(n, vector<int>());
                    queue<int> q;

                    distance[local_in] = 0;
                    q.push(local_in);

                    while (!q.empty()) {
                        int u = q.front(); q.pop();
                        for (int v = 0; v < n; v++) {
                            if (g[u][v]) {
                                if (distance[v] == -1) {
                                    distance[v] = distance[u] + 1;
                                    predecessors[v].push_back(u);
                                    q.push(v);
                                }
                                else if (distance[v] == distance[u] + 1) {
                                    predecessors[v].push_back(u);
                                }
                            }
                        }
                    }

                    if (distance[local_out] == -1) {
                        all_reachable = false;
                    }
                    else {
                        // Reconstruct all shortest paths
                        vector<vector<int>> all_paths;
                        vector<int> path;
                        reconstruct_paths(local_out, local_in, predecessors, path, all_paths);

                        // Since reconstruct_paths adds paths in reverse, reverse them
                        for (auto &p : all_paths) {
                            reverse(p.begin(), p.end());
                        }
                    }
                }
            }
        }
    }

    return all_reachable;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int rows = 4;
    int cols = 4;

    // 1) Generate minimal network
    Graph minimal_graph = generate_minimal_grid_network(rows, cols);
    int n = minimal_graph.size();

    // 2) Initialize reachability Paths
    vector<bitset<MAXN>> Paths(n);
    compute_initial_paths(minimal_graph, Paths);

    // 3) Collect candidate edges
    auto candidates = collect_candidate_internal_edges(rows, cols);
    
    // 4) Attempt to add edges without creating cycles
    double best_result = 0.0;         // Record the current best weighted result
    double current_result = 0.0;      // Record the current weighted result
    Graph best_graph = minimal_graph; // Record the current best graph

    for (int i = 0; i < 1000; i++) {
        random_device rd;
        mt19937 rng(rd());
        shuffle(candidates.begin(), candidates.end(), rng);

        Graph temp_graph = minimal_graph;         // Copy the graph for modifications
        vector<bitset<MAXN>> temp_paths = Paths;      // Copy the reachability paths

        for (auto &edge : candidates) {
            int u = edge.first;
            int v = edge.second;

            if (temp_graph[u][v] == 1) continue;
            if (would_create_cycle(u, v, temp_paths)) continue;
        
            add_edge_and_update_paths(u, v, temp_graph, temp_paths);
        }
        
        reachibility_check(temp_graph, rows, cols, current_result);
        if (current_result > best_result && reachibility_check1(temp_graph, rows, cols) == true) {
            best_result = current_result;
            best_graph = temp_graph;  // Save the best graph
        }
    }

    // Update the final graph to be the best found solution
    minimal_graph = best_graph;

    // 5) Validate results
    cout << "Final Edge Count = " << count_edges(minimal_graph) << "\n";

    // Use Floyd-Warshall to check for cycles
    floyd_warshall_is_acyclic(minimal_graph);

    // 6) Reachability Check with BFS and output path information
    bool reachable = reachibility_check2(minimal_graph, rows, cols, best_result);
    reachibility_check(minimal_graph, rows, cols, best_result);
    if (reachable) {
        cout << "\nAll local_in nodes can reach all required local_out nodes.\n";
    } else {
        cout << "\nSome local_in nodes cannot reach all required local_out nodes.\n";
    }

    cout << "Best result (weighted sum of best paths): " << best_result << endl;

    return 0;
}