#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <tuple>
#include <cmath>      // For using std::log
using namespace std;

using Graph = vector<vector<int>>;

// ----------------- Function to generate a grid network --------------------
//
// Each tile contains 10 sub-nodes representing:
// local_in (0), local_out (1), west_in (2), west_out (3), south_in (4), south_out (5),
// east_in (6), east_out (7), north_in (8), north_out (9)
Graph generate_grid_network(int rows, int cols) {
    int nodes_per_tile = 10;
    int total_nodes = rows * cols * nodes_per_tile;
    Graph graph(total_nodes, vector<int>(total_nodes, 0));

    // Lambda: Returns the node index in the adjacency matrix based on row, col, and sub_node
    auto node_id = [=](int row, int col, int sub_node) -> int {
        return (row * cols + col) * nodes_per_tile + sub_node;
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

            // Internal connections
            graph[local_in][west_out]  = 1;
            graph[local_in][south_out] = 1;
            graph[local_in][east_out]  = 1;
            graph[local_in][north_out] = 1;

            graph[west_in][local_out]  = 1;
            graph[south_in][local_out] = 1;
            graph[east_in][local_out]  = 1;
            graph[north_in][local_out] = 1;

            graph[west_in][north_out]  = 1;
            graph[south_in][north_out] = 1;
            graph[east_in][north_out]  = 1;

            graph[north_in][south_out] = 1;
            graph[east_in][south_out]  = 1;
            graph[west_in][south_out]  = 1;

            graph[west_in][east_out]   = 1;    
            graph[north_in][east_out]  = 1;
            graph[south_in][east_out]  = 1;
            // The following two lines, if commented out, would implement X-Y routing
            graph[east_in][west_out]   = 1;
            // graph[north_in][west_out]  = 1;
            // graph[south_in][west_out]  = 1;

            // Connections with adjacent tiles
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

// ----------------- Function to count the number of edges --------------------
int count_edges(const Graph &graph) {
    int count = 0;
    for (const auto &row : graph)
        for (int v : row)
            count += v;
    return count;
}

// ----------------- Helper function: Map node index to (row, col, node_type) --------------------
tuple<int, int, string> get_node_info(int node, int rows, int cols) {
    int nodes_per_tile = 10;
    int tile_id = node / nodes_per_tile;
    int sub = node % nodes_per_tile;
    int r = tile_id / cols;
    int c = tile_id % cols;
    string type;
    switch(sub) {
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

// ----------------- Function to compute shortest paths and accumulate ln(#paths) --------------------
void find_shortest_paths_and_accumulate_ln(const Graph &graph, int rows, int cols) {
    int nodes_per_tile = 10;
    int n = graph.size();
    double total_ln = 0.0;      // Accumulate the sum of ln(number of shortest paths)
    int valid_pair_count = 0;   // Count the number of reachable (source, destination) pairs

    // For each tile, use local_in as the source node
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            int local_in = (r * cols + c) * nodes_per_tile + 0;  // local_in is sub-node 0

            // Use BFS to compute the shortest path information from local_in to all other nodes
            vector<int> dist(n, -1);
            vector<long long> path_count(n, 0);
            queue<int> q;

            dist[local_in] = 0;
            path_count[local_in] = 1;
            q.push(local_in);

            while (!q.empty()) {
                int u = q.front();
                q.pop();
                for (int v = 0; v < n; v++) {
                    if (graph[u][v]) {
                        if (dist[v] == -1) {  // First visit to v
                            dist[v] = dist[u] + 1;
                            path_count[v] = path_count[u];
                            q.push(v);
                        } else if (dist[v] == dist[u] + 1) {  // Found an alternative shortest path
                            path_count[v] += path_count[u];
                        }
                    }
                }
            }

            // Process local_out (sub-node 1) for all other tiles
            for (int r2 = 0; r2 < rows; ++r2) {
                for (int c2 = 0; c2 < cols; ++c2) {
                    if (r == r2 && c == c2)
                        continue;
                    int local_out = (r2 * cols + c2) * nodes_per_tile + 1;  // local_out is sub-node 1

                    if (dist[local_out] != -1) {
                        // If a path exists, accumulate ln(number of shortest paths)
                        // Note: path_count[local_out] >= 1, so log is safe to compute
                        total_ln += log(static_cast<double>(path_count[local_out]));
                        valid_pair_count++;
                        // Uncomment the following lines to output detailed info for each pair:
                        // cout << "local_in(" << r << "," << c << ") -> local_out(" << r2 << "," << c2 << "): ";
                        // cout << "Shortest path length = " << dist[local_out] << ", ";
                        // cout << "Number of shortest paths = " << path_count[local_out] << endl;
                    } else {
                        // Uncomment the following lines to output when no path is found:
                        // cout << "local_in(" << r << "," << c << ") -> local_out(" << r2 << "," << c2 << "): ";
                        // cout << "No path found!" << endl;
                    }
                }
            }
        }
    }
    cout << "\nThe accumulated sum of ln(number of shortest paths) is: " << total_ln << endl;
    cout << "Number of valid reachable pairs: " << valid_pair_count << endl;
}

// ----------------- Function: Check if the graph is acyclic using Floyd-Warshall --------------------
bool floyd_warshall_is_acyclic(const Graph &g) {
    int n = g.size();
    vector<vector<int>> reachability(n, vector<int>(n, 0));

    // Initialize the reachability matrix
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            reachability[i][j] = g[i][j];

    // Compute the transitive closure
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                reachability[i][j] = reachability[i][j] || (reachability[i][k] && reachability[k][j]);

    // Check for self-loops (cycle detection)
    bool has_cycle = false;
    for (int i = 0; i < n; ++i) {
        if (reachability[i][i] == 1) {
            has_cycle = true;
            break;
        }
    }

    if (has_cycle) {
        // cout << "Cycle detected" << endl;
        return false;
    } else {
        // cout << "No cycle detected" << endl;
        return true;
    }
}

// ----------------- Main function --------------------
int main() {
    // Try various grid sizes (rows and cols)
    int i = 11, j = 11;
    for (int rows = 2; rows < i; rows++) {
        for (int cols = rows; cols < j; cols++) {
            cout << "rows: " << rows << " cols: " << cols;
            Graph graph = generate_grid_network(rows, cols);
            // cout << "Total edges: " << count_edges(graph) << endl;
            floyd_warshall_is_acyclic(graph);
            // Compute shortest path information and output the accumulated ln(#paths)
            find_shortest_paths_and_accumulate_ln(graph, rows, cols);
        }
    }
    return 0;
}