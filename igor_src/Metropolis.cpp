/*
 * metropolis.cpp
 *
 * This file reproduces the behavior of calling IGoR with:
 *   -batch bar -species human -chain beta -align --all
 *
 * It performs V, D, and J alignments for human TCR beta chain sequences.
 */

#include <fcntl.h>
#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
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

namespace {

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
    5,   -14, -14, -14, -14, 2,   -14, 2,   2,   -14, -14, 1,   1,   1,
    0,   -14, 5,   -14, -14, -14, 2,   2,   -14, -14, 2,   1,   -14, 1,
    1,   0,   -14, -14, 5,   -14, 2,   -14, 2,   -14, 2,   -14, 1,   1,
    -14, 1,   0,   -14, -14, -14, 5,   2,   -14, -14, 2,   -14, 2,   1,
    1,   1,   -14, 0,   -14, -14, 2,   2,   1.5, -14, -12, -12, -12, -12,
    1,   1,   -13, -13, 0,   2,   2,   -14, -14, -14, 1.5, -12, -12, -12,
    -12, -13, -13, 1,   1,   0,   -14, 2,   2,   -14, -12, -12, 1.5, -14,
    -12, -12, 1,   -13, -13, 1,   0,   2,   -14, -14, 2,   -12, -12, -14,
    1.5, -12, -12, -13, 1,   1,   -13, 0,   2,   -14, 2,   -14, -12, -12,
    -12, -12, 1.5, -14, -13, 1,   -13, 1,   0,   -14, 2,   -14, 2,   -12,
    -12, -12, -12, -14, 1.5, 1,   -13, 1,   -13, 0,   -14, 1,   1,   1,
    1,   -13, 1,   -13, -13, 1,   0.5, -12, -12, -12, 0,   1,   -14, 1,
    1,   1,   -13, -13, 1,   1,   -13, -12, 0.5, -12, -12, 0,   1,   1,
    -14, 1,   -13, 1,   -13, 1,   -13, 1,   -12, -12, 0.5, -12, 0,   1,
    1,   1,   -14, -13, 1,   1,   -13, 1,   -13, -12, -12, -12, 0.5, 0,
    0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
    0};

/**
 * Singleton class to hold pre-loaded Pgen computation resources.
 * Loads genomic templates and model parameters only once.
 */
class PgenResources {
   public:
    static PgenResources& instance() {
        static PgenResources inst;
        return inst;
    }

    bool is_initialized() const { return initialized_; }
    const string& error_message() const { return error_message_; }

    const Matrix<double>& subst_matrix() const { return subst_matrix_; }
    const vector<pair<string, string>>& v_genomic() const { return v_genomic_; }
    const vector<pair<string, string>>& d_genomic() const { return d_genomic_; }
    const vector<pair<string, string>>& j_genomic() const { return j_genomic_; }
    const Model_Parms& model_parms() const { return model_parms_; }
    const Model_marginals& model_marginals() const { return model_marginals_; }

   private:
    PgenResources() : subst_matrix_(15, 15, SUBST_MATRIX_VALUES) {
        initialized_ = load_resources();
    }

    bool load_resources() {
        string base_path = string(IGOR_DATA_DIR) + "/models/" + PGEN_SPECIES +
                           "/" + PGEN_CHAIN;

        // Load genomic templates
        try {
            v_genomic_ = read_genomic_fasta(base_path +
                                            "/ref_genome/genomicVs.fasta");
        } catch (exception& e) {
            error_message_ =
                string("Failed to read V genomic templates: ") + e.what();
            return false;
        }

        try {
            d_genomic_ = read_genomic_fasta(base_path +
                                            "/ref_genome/genomicDs.fasta");
        } catch (exception& e) {
            error_message_ =
                string("Failed to read D genomic templates: ") + e.what();
            return false;
        }

        try {
            j_genomic_ = read_genomic_fasta(base_path +
                                            "/ref_genome/genomicJs.fasta");
        } catch (exception& e) {
            error_message_ =
                string("Failed to read J genomic templates: ") + e.what();
            return false;
        }

        // Load model parameters
        try {
            model_parms_.read_model_parms(base_path + "/models/model_parms.txt");
        } catch (exception& e) {
            error_message_ =
                string("Failed to read model parameters: ") + e.what();
            return false;
        }

        // Load model marginals
        model_marginals_ = Model_marginals(model_parms_);
        try {
            model_marginals_.txt2marginals(
                base_path + "/models/model_marginals.txt", model_parms_);
        } catch (exception& e) {
            error_message_ =
                string("Failed to read model marginals: ") + e.what();
            return false;
        }

        return true;
    }

    bool initialized_ = false;
    string error_message_;
    Matrix<double> subst_matrix_;
    vector<pair<string, string>> v_genomic_;
    vector<pair<string, string>> d_genomic_;
    vector<pair<string, string>> j_genomic_;
    Model_Parms model_parms_;
    Model_marginals model_marginals_;
};

}  // anonymous namespace

/**
 * Performs alignment equivalent to:
 *   igor -batch bar -species human -chain beta -align --all
 *
 * @param working_directory The working directory path (defaults to /tmp/ if
 * empty)
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
int run_alignment_old(const string& working_directory, const string& sequence) {
    // Configuration matching command line arguments
    const string batchname = "bar_";
    const string species_str = "human";
    const string chain_path_str = "tcr_beta";
    const bool has_D = true;  // TCR beta has D segment

    // Set working directory (default to /tmp/ like main.cpp)
    string cl_path = working_directory.empty() ? "/tmp/" : working_directory;
    if (cl_path.back() != '/') {
        cl_path += "/";
    }
    clog << "Working directory set to: \"" << cl_path << "\"" << endl;

    // Alignment output filenames
    const string v_align_filename = "V_alignments.csv";
    const string d_align_filename = "D_alignments.csv";
    const string j_align_filename = "J_alignments.csv";

    // Alignment parameters (defaults from main.cpp)
    const double v_align_thresh_value = 50.0;
    const double d_align_thresh_value = 15.0;
    const double j_align_thresh_value = 15.0;
    const double v_gap_penalty = 50.0;
    const double d_gap_penalty = 50.0;
    const double j_gap_penalty = 50.0;
    const bool v_best_align_only = true;
    const bool d_best_align_only = false;
    const bool j_best_align_only = true;
    const bool v_best_gene_only = false;
    const bool d_best_gene_only = false;
    const bool j_best_gene_only = false;
    const int v_left_offset_bound = INT16_MIN;
    const int v_right_offset_bound = INT16_MAX;
    const int d_left_offset_bound = INT16_MIN;
    const int d_right_offset_bound = INT16_MAX;
    const int j_left_offset_bound = INT16_MIN;
    const int j_right_offset_bound = INT16_MAX;
    const bool v_reversed_offsets = false;
    const bool d_reversed_offsets = false;
    const bool j_reversed_offsets = false;

    // Substitution matrix (heavy_pen_nuc44 from main.cpp)
    // A,C,G,T,R,Y,K,M,S,W,B,D,H,V,N
    double heavy_pen_nuc44_vect[] = {
        5,   -14, -14, -14, -14, 2,   -14, 2,   2,   -14, -14, 1,   1,   1,
        0,   -14, 5,   -14, -14, -14, 2,   2,   -14, -14, 2,   1,   -14, 1,
        1,   0,   -14, -14, 5,   -14, 2,   -14, 2,   -14, 2,   -14, 1,   1,
        -14, 1,   0,   -14, -14, -14, 5,   2,   -14, -14, 2,   -14, 2,   1,
        1,   1,   -14, 0,   -14, -14, 2,   2,   1.5, -14, -12, -12, -12, -12,
        1,   1,   -13, -13, 0,   2,   2,   -14, -14, -14, 1.5, -12, -12, -12,
        -12, -13, -13, 1,   1,   0,   -14, 2,   2,   -14, -12, -12, 1.5, -14,
        -12, -12, 1,   -13, -13, 1,   0,   2,   -14, -14, 2,   -12, -12, -14,
        1.5, -12, -12, -13, 1,   1,   -13, 0,   2,   -14, 2,   -14, -12, -12,
        -12, -12, 1.5, -14, -13, 1,   -13, 1,   0,   -14, 2,   -14, 2,   -12,
        -12, -12, -12, -14, 1.5, 1,   -13, 1,   -13, 0,   -14, 1,   1,   1,
        1,   -13, 1,   -13, -13, 1,   0.5, -12, -12, -12, 0,   1,   -14, 1,
        1,   1,   -13, -13, 1,   1,   -13, -12, 0.5, -12, -12, 0,   1,   1,
        -14, 1,   -13, 1,   -13, 1,   -13, 1,   -12, -12, 0.5, -12, 0,   1,
        1,   1,   -14, -13, 1,   1,   -13, 1,   -13, -12, -12, -12, 0.5, 0,
        0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
        0};
    Matrix<double> subst_matrix(15, 15, heavy_pen_nuc44_vect);

    // Genomic templates
    vector<pair<string, string>> v_genomic;
    vector<pair<string, string>> d_genomic;
    vector<pair<string, string>> j_genomic;

    // CDR3 anchors
    unordered_map<string, size_t> v_CDR3_anchors;
    unordered_map<string, size_t> j_CDR3_anchors;

    // Read genomic templates for human TCR beta
    clog << "Reading genomic templates for " << species_str << " "
         << chain_path_str << endl;

    try {
        v_genomic = read_genomic_fasta(string(IGOR_DATA_DIR) + "/models/" +
                                       species_str + "/" + chain_path_str +
                                       "/ref_genome/genomicVs.fasta");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading TRB V genomic "
                "templates: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    try {
        d_genomic = read_genomic_fasta(string(IGOR_DATA_DIR) + "/models/" +
                                       species_str + "/" + chain_path_str +
                                       "/ref_genome/genomicDs.fasta");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading TRB D genomic "
                "templates: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    try {
        j_genomic = read_genomic_fasta(string(IGOR_DATA_DIR) + "/models/" +
                                       species_str + "/" + chain_path_str +
                                       "/ref_genome/genomicJs.fasta");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading TRB J genomic "
                "templates: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    clog << "Loaded " << v_genomic.size() << " V genes, " << d_genomic.size()
         << " D genes, " << j_genomic.size() << " J genes" << endl;

    // Read CDR3 anchors
    try {
        v_CDR3_anchors = read_gene_anchors_csv(
            string(IGOR_DATA_DIR) + "/models/" + species_str + "/" +
            chain_path_str + "/ref_genome/V_gene_CDR3_anchors.csv");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading V CDR3 anchors: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    try {
        j_CDR3_anchors = read_gene_anchors_csv(
            string(IGOR_DATA_DIR) + "/models/" + species_str + "/" +
            chain_path_str + "/ref_genome/J_gene_CDR3_anchors.csv");
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading J CDR3 anchors: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Read indexed sequences
    int index = 0;
    vector<pair<const int, const string>> indexed_seqlist = {{index, sequence}};

    // Create directory that will later be used to write alignment files
    system(&("mkdir " + cl_path + "aligns/")[0]);

    // try {
    //     indexed_seqlist = read_indexed_csv(cl_path + "aligns/" + batchname +
    //                                        "indexed_sequences.csv");
    // } catch (exception& e) {
    //     cerr << "[IGoR] ERROR: Exception caught while reading indexed "
    //             "sequences file. "
    //          << "Make sure indexed sequence file has been created using "
    //             "\"-read_seqs\". "
    //          << e.what() << endl;
    //     return EXIT_FAILURE;
    // }

    clog << "Read " << indexed_seqlist.size() << " sequences for alignment"
         << endl;

    // Perform V alignments
    clog << "Performing V alignments...." << endl;
    Aligner v_aligner(subst_matrix, v_gap_penalty, V_gene);
    v_aligner.set_genomic_sequences(v_genomic);
    try {
        v_aligner.align_seqs(
            cl_path + "aligns/" + batchname + v_align_filename, indexed_seqlist,
            v_align_thresh_value, v_best_align_only, v_best_gene_only,
            v_left_offset_bound, v_right_offset_bound, v_reversed_offsets);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught upon aligning V genomic "
                "templates: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Perform D alignments (only if chain has D segment)
    if (has_D) {
        clog << "Performing D alignments...." << endl;
        Aligner d_aligner(subst_matrix, d_gap_penalty, D_gene);
        d_aligner.set_genomic_sequences(d_genomic);
        try {
            d_aligner.align_seqs(
                cl_path + "aligns/" + batchname + d_align_filename,
                indexed_seqlist, d_align_thresh_value, d_best_align_only,
                d_best_gene_only, d_left_offset_bound, d_right_offset_bound,
                d_reversed_offsets);
        } catch (exception& e) {
            cerr << "[IGoR] ERROR: Exception caught upon aligning D genomic "
                    "templates: "
                 << e.what() << endl;
            return EXIT_FAILURE;
        }
    }

    // Perform J alignments
    clog << "Performing J alignments...." << endl;
    Aligner j_aligner(subst_matrix, j_gap_penalty, J_gene);
    j_aligner.set_genomic_sequences(j_genomic);
    try {
        j_aligner.align_seqs(
            cl_path + "aligns/" + batchname + j_align_filename, indexed_seqlist,
            j_align_thresh_value, j_best_align_only, j_best_gene_only,
            j_left_offset_bound, j_right_offset_bound, j_reversed_offsets);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught upon aligning J genomic "
                "templates: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    // Extract CDR3 sequences (b_feature_CDR3 defaults to true in main.cpp)
    clog << "Performing CDR3 sequence extraction ...." << endl;

    unordered_map<
        int, pair<string, unordered_map<Gene_class, vector<Alignment_data>>>>
        sorted_alignments;

    try {
        sorted_alignments = read_alignments_seq_csv_score_range(
            cl_path + "aligns/" + batchname + v_align_filename, V_gene, 55,
            false, indexed_seqlist);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading V alignments "
                "before feature extraction: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    try {
        sorted_alignments = read_alignments_seq_csv_score_range(
            cl_path + "aligns/" + batchname + j_align_filename, J_gene, 10,
            false, indexed_seqlist, sorted_alignments);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught while reading J alignments "
                "before feature extraction: "
             << e.what() << endl;
        return EXIT_FAILURE;
    }

    string flnIndexedCDR3 =
        cl_path + "aligns/" + batchname + "indexed_CDR3s.csv";
    ofstream ofileIndexedCDR3(flnIndexedCDR3);
    ofileIndexedCDR3 << "seq_index;v_anchor;j_anchor;CDR3nt;CDR3aa" << endl;

    ExtractFeatures featureCDR3;
    featureCDR3.load_VJgenomicTemplates(v_genomic, j_genomic);
    featureCDR3.load_VJanchors(v_CDR3_anchors, j_CDR3_anchors);
    featureCDR3.set_sorted_alignments(&sorted_alignments);

    // For each sequence get the CDR3
    for (auto seq_it = indexed_seqlist.begin(); seq_it != indexed_seqlist.end();
         ++seq_it) {
        CDR3SeqData cdr3InputSeq;
        int seq_index = (*seq_it).first;
        cdr3InputSeq = featureCDR3.extractCDR3(seq_index);
        ofileIndexedCDR3 << featureCDR3.generateCDR3_csv_line(cdr3InputSeq)
                         << endl;
    }
    ofileIndexedCDR3.close();

    clog << "Alignment complete. Results written to " << cl_path << "aligns/"
         << batchname << endl;

    return EXIT_SUCCESS;
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
double compute_pgen(const string& working_directory, const string& sequence,
                    bool verbose) {
    // Suppress all output from IGoR library functions if not verbose
    // We need to redirect both C++ streams AND C-level file descriptors
    int stdout_fd = -1;
    int stderr_fd = -1;
    int dev_null_fd = -1;
    streambuf* cout_buf = nullptr;
    streambuf* clog_buf = nullptr;

    if (!verbose) {
        // Flush C++ streams first
        cout.flush();
        clog.flush();
        cerr.flush();
        fflush(stdout);
        fflush(stderr);

        // Save original file descriptors
        stdout_fd = dup(STDOUT_FILENO);
        stderr_fd = dup(STDERR_FILENO);

        // Open /dev/null and redirect stdout/stderr to it
        dev_null_fd = open("/dev/null", O_WRONLY);
        if (dev_null_fd != -1) {
            dup2(dev_null_fd, STDOUT_FILENO);
            dup2(dev_null_fd, STDERR_FILENO);
        }

        // Also redirect C++ streams
        static ofstream null_stream("/dev/null");
        cout_buf = cout.rdbuf(null_stream.rdbuf());
        clog_buf = clog.rdbuf(null_stream.rdbuf());
    }

    // RAII helper to restore streams on exit (including exceptions)
    struct StreamRestorer {
        int stdout_fd;
        int stderr_fd;
        int dev_null_fd;
        streambuf* cout_buf;
        streambuf* clog_buf;
        bool active;
        ~StreamRestorer() {
            if (active) {
                // Flush before restoring
                cout.flush();
                clog.flush();
                fflush(stdout);
                fflush(stderr);

                // Restore C-level file descriptors
                if (stdout_fd != -1) {
                    dup2(stdout_fd, STDOUT_FILENO);
                    close(stdout_fd);
                }
                if (stderr_fd != -1) {
                    dup2(stderr_fd, STDERR_FILENO);
                    close(stderr_fd);
                }
                if (dev_null_fd != -1) {
                    close(dev_null_fd);
                }

                // Restore C++ streams
                if (cout_buf) cout.rdbuf(cout_buf);
                if (clog_buf) clog.rdbuf(clog_buf);
            }
        }
    } restorer{stdout_fd, stderr_fd, dev_null_fd, cout_buf, clog_buf, !verbose};

    // Get pre-loaded resources (loaded only once on first call)
    PgenResources& res = PgenResources::instance();
    if (!res.is_initialized()) {
        cerr << "[IGoR] ERROR: Failed to initialize Pgen resources: "
             << res.error_message() << endl;
        return std::nan("");
    }

    // Set working directory
    string cl_path = working_directory.empty() ? "/tmp/" : working_directory;
    if (cl_path.back() != '/') {
        cl_path += "/";
    }

    // Create indexed sequence list with single sequence
    int index = 0;
    vector<pair<const int, const string>> indexed_seqlist = {{index, sequence}};

    // Create alignment directory
    system(&("mkdir -p " + cl_path + "aligns/")[0]);

    // Perform V alignments
    Aligner v_aligner(res.subst_matrix(), V_GAP_PENALTY, V_gene);
    v_aligner.set_genomic_sequences(res.v_genomic());
    try {
        v_aligner.align_seqs(
            cl_path + "aligns/" + PGEN_BATCHNAME + V_ALIGN_FILENAME,
            indexed_seqlist, V_ALIGN_THRESH, V_BEST_ALIGN_ONLY, V_BEST_GENE_ONLY,
            V_LEFT_OFFSET, V_RIGHT_OFFSET, V_REVERSED_OFFSETS);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught upon aligning V genomic "
                "templates: "
             << e.what() << endl;
        return std::nan("");
    }

    // Perform D alignments
    Aligner d_aligner(res.subst_matrix(), D_GAP_PENALTY, D_gene);
    d_aligner.set_genomic_sequences(res.d_genomic());
    try {
        d_aligner.align_seqs(
            cl_path + "aligns/" + PGEN_BATCHNAME + D_ALIGN_FILENAME,
            indexed_seqlist, D_ALIGN_THRESH, D_BEST_ALIGN_ONLY, D_BEST_GENE_ONLY,
            D_LEFT_OFFSET, D_RIGHT_OFFSET, D_REVERSED_OFFSETS);
    } catch (exception& e) {
        cerr << "[IGoR] ERROR: Exception caught upon aligning D genomic "
                "templates: "
             << e.what() << endl;
        return std::nan("");
    }

    // Perform J alignments
    Aligner j_aligner(res.subst_matrix(), J_GAP_PENALTY, J_gene);
    j_aligner.set_genomic_sequences(res.j_genomic());
    try {
        j_aligner.align_seqs(
            cl_path + "aligns/" + PGEN_BATCHNAME + J_ALIGN_FILENAME,
            indexed_seqlist, J_ALIGN_THRESH, J_BEST_ALIGN_ONLY, J_BEST_GENE_ONLY,
            J_LEFT_OFFSET, J_RIGHT_OFFSET, J_REVERSED_OFFSETS);
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
    GenModel genmodel(res.model_parms(), res.model_marginals(), cl_counters_list);

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

int main(int argc, char* argv[]) {
    string working_dir =
        "/Users/alexanderbonnet/code/statbiophys-technical-test/data/1";
    if (argc > 1) {
        working_dir = argv[1];
    }

    string sequence =
        "GACGCTGGAGTCACCCAAAGTCCCACACACCTGATCAAAACGAGAGGACAGCAAGTGACTCTGAGATGCT"
        "CTCCTAAGTCTGGGCATGACACTGTGTCCTGGTACCAACAGGCCCTGGGTCAGGGGCCCCAGTTTATCTT"
        "TCAGTATTATGAGGAGGAAGAGAGACAGAGAGGCAACTTCCCTGATCGATTCTCAGGTCACCAGTTCCCT"
        "AACTATAGCTCTGAGCTGAATGTGAACGCCTTGTTGCTGGGGGACTCGGCCCTCTATCTCTGTGCCAGCA"
        "GCTTGGGCTCAGGGAATGTTTCAGGGAAACACCATTTATTATGGAGAGGGAAGTTGGCTCACTGTTGTA"
        "G";

    // run_alignment_old(working_dir, sequence);
    // cout << "oasihfoalfhal";

    double pgen = compute_pgen(working_dir, sequence, false);
    cout << pgen;
    // run_alignment(working_dir, sequence);

    return EXIT_SUCCESS;
}
