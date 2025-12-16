#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "../config.h"
#include "Aligner.h"
#include "CDR3SeqData.h"
#include "ExtractFeatures.h"
#include "GenModel.h"
#include "Model_Parms.h"
#include "Model_marginals.h"
#include "Pgencounter.h"
#include "Utils.h"

using namespace std;

const std::array<char, 4> NUCLEOTIDES = {'A', 'G', 'T', 'C'};

// =============================================================================
// Pgen computation constants and pre-loaded data
// =============================================================================

// Configuration constants
const string PGEN_BATCHNAME = "bar_";
const string PGEN_SPECIES = "human";
const string PGEN_CHAIN = "tcr_beta";

// Alignment filenames
const string V_ALIGN_FILENAME = "V_alignments.csv";
const string D_ALIGN_FILENAME = "D_alignments.csv";
const string J_ALIGN_FILENAME = "J_alignments.csv";

// Alignment parameters
const double V_ALIGN_THRESH = 50.0;
const double D_ALIGN_THRESH = 15.0;
const double J_ALIGN_THRESH = 15.0;
const double V_GAP_PENALTY = 50.0;
const double D_GAP_PENALTY = 50.0;
const double J_GAP_PENALTY = 50.0;
const bool V_BEST_ALIGN_ONLY = true;
const bool D_BEST_ALIGN_ONLY = false;
const bool J_BEST_ALIGN_ONLY = true;
const bool V_BEST_GENE_ONLY = false;
const bool D_BEST_GENE_ONLY = false;
const bool J_BEST_GENE_ONLY = false;
const int V_LEFT_OFFSET = INT16_MIN;
const int V_RIGHT_OFFSET = INT16_MAX;
const int D_LEFT_OFFSET = INT16_MIN;
const int D_RIGHT_OFFSET = INT16_MAX;
const int J_LEFT_OFFSET = INT16_MIN;
const int J_RIGHT_OFFSET = INT16_MAX;
const bool V_REVERSED_OFFSETS = false;
const bool D_REVERSED_OFFSETS = false;
const bool J_REVERSED_OFFSETS = false;

// Evaluate parameters
const double LIKELIHOOD_THRESH = 1e-60;
const double PROBA_THRESHOLD_RATIO = 1e-5;
const bool VITERBI_EVALUATE = false;

// Substitution matrix values (heavy_pen_nuc44 from main.cpp)
// A,C,G,T,R,Y,K,M,S,W,B,D,H,V,N
double SUBST_MATRIX_VALUES[] = {
    5,   -14, -14, -14, -14, 2,   -14, 2,   2,   -14, -14, 1,   1,   1,   0,
    -14, 5,   -14, -14, -14, 2,   2,   -14, -14, 2,   1,   -14, 1,   1,   0,
    -14, -14, 5,   -14, 2,   -14, 2,   -14, 2,   -14, 1,   1,   -14, 1,   0,
    -14, -14, -14, 5,   2,   -14, -14, 2,   -14, 2,   1,   1,   1,   -14, 0,
    -14, -14, 2,   2,   1.5, -14, -12, -12, -12, -12, 1,   1,   -13, -13, 0,
    2,   2,   -14, -14, -14, 1.5, -12, -12, -12, -12, -13, -13, 1,   1,   0,
    -14, 2,   2,   -14, -12, -12, 1.5, -14, -12, -12, 1,   -13, -13, 1,   0,
    2,   -14, -14, 2,   -12, -12, -14, 1.5, -12, -12, -13, 1,   1,   -13, 0,
    2,   -14, 2,   -14, -12, -12, -12, -12, 1.5, -14, -13, 1,   -13, 1,   0,
    -14, 2,   -14, 2,   -12, -12, -12, -12, -14, 1.5, 1,   -13, 1,   -13, 0,
    -14, 1,   1,   1,   1,   -13, 1,   -13, -13, 1,   0.5, -12, -12, -12, 0,
    1,   -14, 1,   1,   1,   -13, -13, 1,   1,   -13, -12, 0.5, -12, -12, 0,
    1,   1,   -14, 1,   -13, 1,   -13, 1,   -13, 1,   -12, -12, 0.5, -12, 0,
    1,   1,   1,   -14, -13, 1,   1,   -13, 1,   -13, -12, -12, -12, 0.5, 0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0};

// A RAII-based class to suppress stdout/stderr and cout/clog.
struct SilentMode {
    SilentMode() {
        // Flush existing streams
        cout.flush();
        clog.flush();
        cerr.flush();
        fflush(stdout);
        fflush(stderr);

        // Save original file descriptors and stream buffers
        cout_buf = std::cout.rdbuf();
        clog_buf = std::clog.rdbuf();
        stdout_fd = dup(STDOUT_FILENO);
        stderr_fd = dup(STDERR_FILENO);

        // Redirect C++ streams
        dev_null_stream.open("/dev/null");
        std::cout.rdbuf(dev_null_stream.rdbuf());
        std::clog.rdbuf(dev_null_stream.rdbuf());

        // Redirect C file descriptors
        dev_null_fd = open("/dev/null", O_WRONLY);
        dup2(dev_null_fd, STDOUT_FILENO);
        dup2(dev_null_fd, STDERR_FILENO);
    }

    ~SilentMode() {
        // Flush before restoring
        cout.flush();
        clog.flush();
        fflush(stdout);
        fflush(stderr);

        // Restore C++ streams
        std::cout.rdbuf(cout_buf);
        std::clog.rdbuf(clog_buf);
        dev_null_stream.close();

        // Restore C file descriptors
        dup2(stdout_fd, STDOUT_FILENO);
        dup2(stderr_fd, STDERR_FILENO);

        // Close saved file descriptors
        close(stdout_fd);
        close(stderr_fd);
        close(dev_null_fd);
    }

   private:
    std::streambuf* cout_buf;
    std::streambuf* clog_buf;
    int stdout_fd;
    int stderr_fd;
    int dev_null_fd;
    std::ofstream dev_null_stream;
};
struct Resources {
    vector<pair<string, string>> v_genomic;
    vector<pair<string, string>> d_genomic;
    vector<pair<string, string>> j_genomic;
    Model_Parms model_params;
    Model_marginals model_marginals;
    Matrix<double> subst_matrix;
};

// load resources only once instead of every time we need to coompute pgen
Resources load_resources(bool verbose) {
    unique_ptr<SilentMode> silent_mode;
    if (!verbose) {
        silent_mode = make_unique<SilentMode>();
    }

    vector<pair<string, string>> v_genomic;
    vector<pair<string, string>> d_genomic;
    vector<pair<string, string>> j_genomic;

    string base_path =
        string(IGOR_DATA_DIR) + "/models/" + PGEN_SPECIES + "/" + PGEN_CHAIN;

    // Read genomic templates for human TCR beta
    try {
        v_genomic =
            read_genomic_fasta(base_path + "/ref_genome/genomicVs.fasta");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading TRB V genomic "
                "templates: "
             << e.what() << endl;
    }

    try {
        d_genomic =
            read_genomic_fasta(base_path + "/ref_genome/genomicDs.fasta");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading TRB D genomic "
                "templates: "
             << e.what() << endl;
    }

    try {
        j_genomic =
            read_genomic_fasta(base_path + "/ref_genome/genomicJs.fasta");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading TRB J genomic "
                "templates: "
             << e.what() << endl;
    }

    Model_Parms model_params;
    try {
        model_params.read_model_parms(base_path + "/models/model_parms.txt");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading model "
                "parameters: "
             << e.what() << endl;
    }

    Model_marginals model_marginals(model_params);
    try {
        model_marginals.txt2marginals(base_path + "/models/model_marginals.txt",
                                      model_params);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading model "
                "marginals: "
             << e.what() << endl;
    }

    Matrix<double> subst_matrix(15, 15, SUBST_MATRIX_VALUES);

    Resources result = {v_genomic,    d_genomic,       j_genomic,
                        model_params, model_marginals, subst_matrix};

    return result;
};

/**
 * Computes the generation probability (Pgen) for a sequence.
 * Replicates the behavior of:
 *   igor -batch bar -species human -chain beta -evaluate -output --Pgen
 *
 * @param working_directory The working directory path for intermediate files
 * @param sequence The nucleotide sequence to evaluate
 * @return The estimated generation probability (Pgen), or NaN if evaluation
 * fails
 */

// ... (rest of the includes)

double compute_pgen(const string& workdir, const string& sequence,
                    Resources resources, bool verbose) {
    unique_ptr<SilentMode> silent_mode;
    if (!verbose) {
        silent_mode = make_unique<SilentMode>();
    }

    // Set working directory
    string cl_path = workdir.empty() ? "/tmp/" : workdir;
    if (cl_path.back() != '/') {
        cl_path += "/";
    }

    // Create indexed sequence list with single sequence
    int index = 0;
    vector<pair<const int, const string>> indexed_seqlist = {{index, sequence}};

    // Create alignment directory
    system(&("mkdir -p " + cl_path + "aligns/")[0]);

    // Perform V alignments
    Aligner v_aligner(resources.subst_matrix, V_GAP_PENALTY, V_gene);
    v_aligner.set_genomic_sequences(resources.v_genomic);
    try {
        v_aligner.align_seqs(
            cl_path + "aligns/" + PGEN_BATCHNAME + V_ALIGN_FILENAME,
            indexed_seqlist, V_ALIGN_THRESH, V_BEST_ALIGN_ONLY,
            V_BEST_GENE_ONLY, V_LEFT_OFFSET, V_RIGHT_OFFSET,
            V_REVERSED_OFFSETS);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught upon aligning V genomic "
                "templates: "
             << e.what() << endl;
        return std::nan("");
    }

    // Perform D alignments
    Aligner d_aligner(resources.subst_matrix, D_GAP_PENALTY, D_gene);
    d_aligner.set_genomic_sequences(resources.d_genomic);
    try {
        d_aligner.align_seqs(
            cl_path + "aligns/" + PGEN_BATCHNAME + D_ALIGN_FILENAME,
            indexed_seqlist, D_ALIGN_THRESH, D_BEST_ALIGN_ONLY,
            D_BEST_GENE_ONLY, D_LEFT_OFFSET, D_RIGHT_OFFSET,
            D_REVERSED_OFFSETS);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught upon aligning D genomic "
                "templates: "
             << e.what() << endl;
        return std::nan("");
    }

    // Perform J alignments
    Aligner j_aligner(resources.subst_matrix, J_GAP_PENALTY, J_gene);
    j_aligner.set_genomic_sequences(resources.j_genomic);
    try {
        j_aligner.align_seqs(
            cl_path + "aligns/" + PGEN_BATCHNAME + J_ALIGN_FILENAME,
            indexed_seqlist, J_ALIGN_THRESH, J_BEST_ALIGN_ONLY,
            J_BEST_GENE_ONLY, J_LEFT_OFFSET, J_RIGHT_OFFSET,
            J_REVERSED_OFFSETS);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught upon aligning J genomic "
                "templates: "
             << e.what() << endl;
        return std::nan("");
    }

    // Create Pgen counter (output_Pgen_estimator_only = true for evaluate mode)
    string evaluate_path = cl_path + PGEN_BATCHNAME + "evaluate/";
    system(&("mkdir -p " + evaluate_path + "output")[0]);

    map<size_t, shared_ptr<Counter>> cl_counters_list;
    shared_ptr<Counter> pgen_counter_ptr(
        new Pgen_counter(evaluate_path + "output/", true));
    cl_counters_list.emplace(cl_counters_list.size(), pgen_counter_ptr);

    // Create GenModel with pre-loaded model and counters
    GenModel genmodel(resources.model_params, resources.model_marginals,
                      cl_counters_list);

    // Read alignments back
    unordered_map<
        int, pair<string, unordered_map<Gene_class, vector<Alignment_data>>>>
        sorted_alignments;
    try {
        sorted_alignments = read_alignments_seq_csv_score_range(
            cl_path + "aligns/" + PGEN_BATCHNAME + V_ALIGN_FILENAME, V_gene, 55,
            false, indexed_seqlist);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading V alignments: "
             << e.what() << endl;
        return std::nan("");
    }

    try {
        sorted_alignments = read_alignments_seq_csv_score_range(
            cl_path + "aligns/" + PGEN_BATCHNAME + D_ALIGN_FILENAME, D_gene, 35,
            false, indexed_seqlist, sorted_alignments);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading D alignments: "
             << e.what() << endl;
        return std::nan("");
    }

    try {
        sorted_alignments = read_alignments_seq_csv_score_range(
            cl_path + "aligns/" + PGEN_BATCHNAME + J_ALIGN_FILENAME, J_gene, 10,
            false, indexed_seqlist, sorted_alignments);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading J alignments: "
             << e.what() << endl;
        return std::nan("");
    }

    // Convert to vector format required by infer_model
    vector<
        tuple<int, string, unordered_map<Gene_class, vector<Alignment_data>>>>
        sorted_alignments_vec = map2vect(sorted_alignments);

    // Check if we have any alignments to evaluate
    if (sorted_alignments_vec.empty()) {
        cerr << "[IGoR] WARNING: No valid alignments found for sequence"
             << endl;
        return std::nan("");
    }

    // Run evaluation (1 iteration, fast_iter=false for evaluate mode)
    genmodel.infer_model(sorted_alignments_vec, 1, evaluate_path, false,
                         LIKELIHOOD_THRESH, VITERBI_EVALUATE,
                         PROBA_THRESHOLD_RATIO);

    // Read the Pgen result from the output file
    string pgen_output_file = evaluate_path + "output/Pgen_counts.csv";
    ifstream pgen_file(pgen_output_file);
    if (!pgen_file.is_open()) {
        cerr << "[IGoR] ERROR: Could not open Pgen output file: "
             << pgen_output_file << endl;
        return std::nan("");
    }

    string line;
    double pgen_estimate = std::nan("");

    // Skip header
    if (getline(pgen_file, line)) {
        // Read first data line (seq_index;Pgen_estimate)
        if (getline(pgen_file, line)) {
            size_t semicolon_pos = line.find(';');
            if (semicolon_pos != string::npos) {
                string pgen_str = line.substr(semicolon_pos + 1);
                try {
                    pgen_estimate = stod(pgen_str);
                } catch (exception& e) {
                    cerr << "[IGoR] ERROR: Could not parse Pgen value: "
                         << pgen_str << endl;
                }
            }
        }
    }

    pgen_file.close();
    return pgen_estimate;
}

int hamming_distance(const std::string& s1, const std::string& s2) {
    if (s1.length() != s2.length()) {
        throw std::invalid_argument("Strings must have identical length.");
    }

    int distance = 0;
    for (size_t i = 0; i < s1.length(); ++i) {
        if (s1[i] != s2[i]) {
            ++distance;
        }
    }
    return distance;
}

char choose_different_nucleotide(char current, std::mt19937& rng) {
    std::array<char, 3> choices;
    int idx = 0;
    for (char nucleotide : NUCLEOTIDES) {
        if (nucleotide != current) {
            choices[idx++] = nucleotide;
        }
    }
    std::uniform_int_distribution<int> dist(0, 2);
    return choices[dist(rng)];
}

std::string mutate(const std::string& sequence, int left, int right,
                   std::mt19937& rng) {
    if (right >= 0) {
        throw std::invalid_argument("'right' should be a negative integer.");
    }

    int adjusted_right = static_cast<int>(sequence.length()) + right;

    if (left >= adjusted_right) {
        throw std::invalid_argument(
            "'left' needs to be strictly inferior to 'right'.");
    }

    std::uniform_int_distribution<int> pos_dist(left, adjusted_right);
    int position = pos_dist(rng);

    char current_nucleotide = sequence[position];
    char mutation = choose_different_nucleotide(current_nucleotide, rng);

    std::string result = sequence;
    result[position] = mutation;
    return result;
}

string double_to_string(double value, int precision = 5) {
    stringstream ss;
    ss << std::scientific << std::setprecision(precision) << value;
    return ss.str();
}
struct StepMetadata {
    string state;
    int distance;
    double pgen;
    int step;

    string to_string() const {
        return state + "," + std::to_string(distance) + "," +
               double_to_string(pgen) + "," + std::to_string(step);
    }
};

void metropolis(const string& workdir, const string& sequence, int num_samples,
                int seed, std::pair<int, int> mutation_region, int buffer_size,
                bool overwrite) {
    // Setup random number generation
    std::mt19937 rng(seed);

    // Setup working directory
    struct stat info;
    bool dir_exists =
        (stat(workdir.c_str(), &info) == 0 && (info.st_mode & S_IFDIR));

    if (dir_exists && overwrite) {
        string command = "rm -rf " + workdir;
        system(command.c_str());
    }
    string mkdir_command = "mkdir -p " + workdir;
    system(mkdir_command.c_str());

    Resources resources = load_resources(false);

    // Initial step
    double current_prob = compute_pgen(workdir, sequence, resources, false);
    if (std::isnan(current_prob)) {
        cerr << "[IGoR] ERROR: Initial Pgen is NaN, aborting." << endl;
        return;
    }

    vector<StepMetadata> metadata_buffer;
    metadata_buffer.push_back({sequence, 0, current_prob, 0});

    ofstream outfile(workdir + "/samples.csv");
    outfile << "state,distance,pgen,step\n";

    string current_state = sequence;
    int num_accepted = 0;

    cout << "[IGoR] Starting Metropolis sampling..." << endl;

    for (int step = 1; step <= num_samples; ++step) {
        string proposal_state = mutate(current_state, mutation_region.first,
                                       mutation_region.second, rng);

        double proposal_prob =
            compute_pgen(workdir, proposal_state, resources, false);

        if (std::isnan(proposal_prob)) {
            continue;
        }

        double acceptance_prob = std::min(1.0, proposal_prob / current_prob);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

        if (uniform_dist(rng) < acceptance_prob) {
            current_state = proposal_state;
            current_prob = proposal_prob;
            num_accepted++;
            int dist = hamming_distance(sequence, proposal_state);
            metadata_buffer.push_back(
                {current_state, dist, current_prob, step});
        }

        if (metadata_buffer.size() >= buffer_size) {
            for (const auto& meta : metadata_buffer) {
                outfile << meta.to_string() << "\n";
            }
            metadata_buffer.clear();
        }

        if (step % 100 == 0 || step == num_samples) {
            double acceptance_ratio =
                (step > 0) ? static_cast<double>(num_accepted) / step : 0.0;
            cout << "[IGoR] Step " << step << "/" << num_samples
                 << " | Acceptance ratio: " << acceptance_ratio
                 << " | Current Pgen: " << current_prob << endl;
        }
    }

    // Write any remaining data in the buffer
    for (const auto& meta : metadata_buffer) {
        outfile << meta.to_string() << "\n";
    }

    cout << "[IGoR] Metropolis sampling finished." << endl;
}

int main(int argc, char* argv[]) {
    string workdir =
        "/Users/alexanderbonnet/code/statbiophys-technical-test/data";
    if (argc > 1) {
        workdir = argv[1];
    }

    string sequence =
        "GACGCTGGAGTCACCCAAAGTCCCACACACCTGATCAAAACGAGAGGACAGCAAGTGACTCTGAGATGCT"
        "CTCCTAAGTCTGGGCATGACACTGTGTCCTGGTACCAACAGGCCCTGGGTCAGGGGCCCCAGTTTATCTT"
        "TCAGTATTATGAGGAGGAAGAGAGACAGAGAGGCAACTTCCCTGATCGATTCTCAGGTCACCAGTTCCCT"
        "AACTATAGCTCTGAGCTGAATGTGAACGCCTTGTTGCTGGGGGACTCGGCCCTCTATCTCTGAAATCGCA"
        "GCTTGGAAGGCGGGAAGGAGTGGGGGAAACACCGTGTACTATGGAGAGGGAAGTTGGCTCACTGTTGTA"
        "G";

    // Resources resources = load_resources();
    // double result = compute_pgen(workdir, sequence, resources, true);

    // cout << result;

    int num_samples = 200;
    int seed = 42;
    std::pair<int, int> mutation_region = {270, -30};
    int buffer_size = 5;
    bool overwrite = true;

    metropolis(workdir, sequence, num_samples, seed, mutation_region,
               buffer_size, overwrite);

    return EXIT_SUCCESS;
}
