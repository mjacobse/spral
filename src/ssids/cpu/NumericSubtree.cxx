/* Copyright 2016 The Science and Technology Facilities Council (STFC)
 *
 * Authors: Jonathan Hogg (STFC)
 *
 * IMPORTANT: This file is NOT licenced under the BSD licence. If you wish to
 * licence this code, please contact STFC via hsl@stfc.ac.uk
 * (We are currently deciding what licence to release this code under if it
 * proves to be useful beyond our own academic experiments)
 *
 */
#include "NumericSubtree.hxx"

#include <cstdio>

#include "StackAllocator.hxx"
#include "NumericSubtree.hxx"
#include "SmallLeafSubtree.hxx"

using namespace spral::ssids::cpu;

/////////////////////////////////////////////////////////////////////////////
// anonymous namespace
namespace {

/* FIXME: remove post debug */
template<typename T>
void print_node(bool posdef, int m, int n, int nelim, const int *perm, const int *rlist, const T *lcol, const T*d) {
   for(int i=0; i<m; i++) {
      if(i<n) printf("%d%s:", perm[i], (i<nelim)?"X":"D");
      else    printf("%d:", rlist[i-n]);
      for(int j=0; j<n; j++) printf(" %10.2e", lcol[j*m+i]);
      if(!posdef && i<nelim) printf("  d: %10.2e %10.2e\n", d[2*i+0], d[2*i+1]);
      else printf("\n");
   }
}

/* FIXME: remove post debug */
template<typename T>
void print_factors(
      bool posdef,
      int nnodes,
      SymbolicSubtree const& symb,
      struct cpu_node_data<T> *const nodes
      ) {
   for(int node=0; node<nnodes; node++) {
      printf("== Node %d ==\n", node);
      int m = symb[node].nrow + nodes[node].ndelay_in;
      int n = symb[node].ncol + nodes[node].ndelay_in;
      const int *rptr = &symb[node].rlist[ symb[node].ncol ];
      print_node(posdef, m, n, nodes[node].nelim, nodes[node].perm,
            rptr, nodes[node].lcol, &nodes[node].lcol[m*n]);
   }
}

} /* end of anon namespace */
//////////////////////////////////////////////////////////////////////////

extern "C"
void* spral_ssids_cpu_create_num_subtree_dbl(bool posdef, void const* symbolic_subtree_ptr, int nnodes, struct cpu_node_data<double>* nodes) {
   const int BLOCK_SIZE = 16;
   const int PAGE_SIZE = 16384;
   typedef double T;

   auto const& symbolic_subtree = *static_cast<SymbolicSubtree const*>(symbolic_subtree_ptr);

   return (posdef) ? (void*) new NumericSubtree
                     <true, BLOCK_SIZE, T, StackAllocator<PAGE_SIZE>>
                     (symbolic_subtree, nnodes, nodes)
                   : (void*) new NumericSubtree
                     <false, BLOCK_SIZE, T, StackAllocator<PAGE_SIZE>>
                     (symbolic_subtree, nnodes, nodes);
}

extern "C"
void spral_ssids_cpu_destroy_num_subtree_dbl(bool posdef, void* target) {
   const int BLOCK_SIZE = 16;
   const int PAGE_SIZE = 16384;
   typedef double T;

   if(!target) return;

   if(posdef) {
      auto *subtree = static_cast<NumericSubtree<true, BLOCK_SIZE, T, StackAllocator<PAGE_SIZE>>*>(target);
      delete subtree;
   } else {
      auto *subtree = static_cast<NumericSubtree<false, BLOCK_SIZE, T, StackAllocator<PAGE_SIZE>>*>(target);
      delete subtree;
   }
}

/* Double precision wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_factor_dbl(
      bool posdef,     // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void* subtree_ptr,// pointer to relevant type of NumericSubtree
      int n,            // Maximum row index (+1)
      int nnodes,       // Number of nodes in assembly tree
      struct cpu_node_data<double> *const nodes, // Data structure for node information
      const double *const aval, // Values of A
      const double *const scaling, // Scaling vector (NULL if none)
      void *const alloc,      // Pointer to Fortran allocator structure
      const struct cpu_factor_options *const options, // Options in
      struct cpu_factor_stats *const stats // Info out
      ) {

   typedef double T;
   const int BLOCK_SIZE = 16;
   const int PAGE_SIZE = 16384;

   // Initialize
   StackAllocator<PAGE_SIZE> stalloc_odd, stalloc_even;
   Workspace<T> work(PAGE_SIZE);
   int *map = new int[n+1];
   stats->flag = SSIDS_SUCCESS;
   stats->num_delay = 0;
   stats->num_neg = 0;
   stats->num_two = 0;
   stats->num_zero = 0;
   stats->maxfront = 0;
   for(int i=0; i<5; i++) stats->elim_at_pass[i] = 0;
   for(int i=0; i<5; i++) stats->elim_at_itr[i] = 0;

   // Call factorization routine
   SymbolicSubtree const* symbolic_subtree;
   if(posdef) { // Converting from runtime to compile time posdef value
      auto &subtree =
         *static_cast<NumericSubtree<true, BLOCK_SIZE, T, StackAllocator<PAGE_SIZE>>*>(subtree_ptr);
      try {
         subtree.factor(aval, scaling, alloc, stalloc_odd, stalloc_even, work, map, options, stats);
      } catch(NotPosDefError npde) {
         stats->flag = SSIDS_ERROR_NOT_POS_DEF;
      }
      symbolic_subtree = &subtree.get_symbolic_subtree();
   } else {
      auto &subtree =
         *static_cast<NumericSubtree<false, BLOCK_SIZE, T, StackAllocator<PAGE_SIZE>>*>(subtree_ptr);
      subtree.factor(aval, scaling, alloc, stalloc_odd, stalloc_even, work, map, options, stats);
      symbolic_subtree = &subtree.get_symbolic_subtree();
   }

   // Release resources
   delete[] map;

   // FIXME: Remove when done with debug
   if(options->print_level > 9999) {
      printf("Final factors:\n");
      print_factors<double>(posdef, nnodes, *symbolic_subtree, nodes);
   }
}

/* Double precision wrapper around templated routines */
extern "C"
void spral_ssids_cpu_subtree_solve_fwd_dbl(
      bool posdef,      // If true, performs A=LL^T, if false do pivoted A=LDL^T
      void* subtree_ptr,// pointer to relevant type of NumericSubtree
      int nrhs,         // number of right-hand sides
      double* x,        // ldx x nrhs array of right-hand sides
      int ldx           // leading dimension of x
      ) {

   typedef double T;
   const int BLOCK_SIZE = 16;
   const int PAGE_SIZE = 16384;

   // Call factorization routine
   if(posdef) { // Converting from runtime to compile time posdef value
      auto &subtree =
         *static_cast<NumericSubtree<true, BLOCK_SIZE, T, StackAllocator<PAGE_SIZE>>*>(subtree_ptr);
      subtree.solve_fwd(nrhs, x, ldx);
   } else {
      auto &subtree =
         *static_cast<NumericSubtree<false, BLOCK_SIZE, T, StackAllocator<PAGE_SIZE>>*>(subtree_ptr);
      subtree.solve_fwd(nrhs, x, ldx);
   }
}
