/*
 * metropolis.cpp
 *
 * This file reproduces the behavior of calling IGoR with:
 *   -batch bar -species human -chain beta -align --all
 *
 * It performs V, D, and J alignments for human TCR beta chain sequences.
 */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../config.h"
#include "Aligner.h"
#include "CDR3SeqData.h"
#include "ExtractFeatures.h"
#include "Utils.h"

using namespace std;

const std::array<char, 4> NUCLEOTIDES = {'A', 'G', 'T', 'C'};

/**
 * Performs alignment equivalent to:
 *   igor -batch bar -species human -chain beta -align --all
 *
 * @param working_directory The working directory path (defaults to /tmp/ if
 * empty)
 * @return EXIT_SUCCESS on success, EXIT_FAILURE on error
 */
int run_alignment(const string& working_directory, const string& sequence) {
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
        "GCTTGGGCTCAGGGTATGTTTCAGGGAAACACCATATATTTTGGAGAGGGAAGTTGGCTCACTGTTGTA"
        "G";
    run_alignment(working_dir, sequence);

    return EXIT_SUCCESS;
}
