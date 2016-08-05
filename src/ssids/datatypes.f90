module spral_ssids_datatypes
!$ use omp_lib
   use, intrinsic :: iso_c_binding
   use spral_scaling, only : auction_options, auction_inform
   implicit none

   private
   public :: smalloc_type, stack_mem_type, stack_type, thread_stats, &
      real_ptr_type, ssids_options, node_type

   integer, parameter, public :: wp = C_DOUBLE
   integer, parameter, public :: long = selected_int_kind(18)

   real(wp), parameter, public :: one = 1.0_wp
   real(wp), parameter, public :: zero = 0.0_wp

   integer, parameter, public :: nemin_default = 8 ! node amalgamation parameter

   ! Success flag
   integer, parameter, public :: SSIDS_SUCCESS                 = 0

   ! Error flags
   integer, parameter, public :: SSIDS_ERROR_CALL_SEQUENCE     = -1
   integer, parameter, public :: SSIDS_ERROR_A_N_OOR           = -2
   integer, parameter, public :: SSIDS_ERROR_A_PTR             = -3
   integer, parameter, public :: SSIDS_ERROR_A_ALL_OOR         = -4
   integer, parameter, public :: SSIDS_ERROR_SINGULAR          = -5
   integer, parameter, public :: SSIDS_ERROR_NOT_POS_DEF       = -6
   integer, parameter, public :: SSIDS_ERROR_PTR_ROW           = -7
   integer, parameter, public :: SSIDS_ERROR_ORDER             = -8
   integer, parameter, public :: SSIDS_ERROR_VAL               = -9
   integer, parameter, public :: SSIDS_ERROR_X_SIZE            = -10
   integer, parameter, public :: SSIDS_ERROR_JOB_OOR           = -11
   integer, parameter, public :: SSIDS_ERROR_PRESOLVE_INCOMPAT = -12
   integer, parameter, public :: SSIDS_ERROR_NOT_LLT           = -13
   integer, parameter, public :: SSIDS_ERROR_NOT_LDLT          = -14
   integer, parameter, public :: SSIDS_ERROR_NO_SAVED_SCALING  = -15
   integer, parameter, public :: SSIDS_ERROR_ALLOCATION        = -50
   integer, parameter, public :: SSIDS_ERROR_CUDA_UNKNOWN      = -51
   integer, parameter, public :: SSIDS_ERROR_CUBLAS_UNKNOWN    = -52
   integer, parameter, public :: SSIDS_ERROR_UNIMPLEMENTED     = -98
   integer, parameter, public :: SSIDS_ERROR_UNKNOWN           = -99

   ! warning flags
   integer, parameter, public :: SSIDS_WARNING_IDX_OOR          = 1
   integer, parameter, public :: SSIDS_WARNING_DUP_IDX          = 2
   integer, parameter, public :: SSIDS_WARNING_DUP_AND_OOR      = 3
   integer, parameter, public :: SSIDS_WARNING_MISSING_DIAGONAL = 4
   integer, parameter, public :: SSIDS_WARNING_MISS_DIAG_OORDUP = 5
   integer, parameter, public :: SSIDS_WARNING_ANAL_SINGULAR    = 6
   integer, parameter, public :: SSIDS_WARNING_FACT_SINGULAR    = 7
   integer, parameter, public :: SSIDS_WARNING_MATCH_ORD_NO_SCALE=8

   ! solve job values
   integer, parameter, public :: SSIDS_SOLVE_JOB_ALL     = 0 !PLD(PL)^TX = B
   integer, parameter, public :: SSIDS_SOLVE_JOB_FWD     = 1 !PLX = B
   integer, parameter, public :: SSIDS_SOLVE_JOB_DIAG    = 2 !DX = B (indef)
   integer, parameter, public :: SSIDS_SOLVE_JOB_BWD     = 3 !(PL)^TX = B
   integer, parameter, public :: SSIDS_SOLVE_JOB_DIAG_BWD= 4 !D(PL)^TX=B (indef)

   ! NB: the below must match enum pivot_method in cpu/cpu_iface.hxx
   integer, parameter, public :: PIVOT_METHOD_APP_AGGRESIVE = 1
   integer, parameter, public :: PIVOT_METHOD_APP_BLOCK     = 2
   integer, parameter, public :: PIVOT_METHOD_TPP           = 3

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   ! Note: below smalloc etc. types can't be in spral_ssids_alloc module as
   ! they are used as components of later datatypes.

   ! Type for custom allocator
   ! Used to aggregate many small allocations by doing a single big allocation
   ! and chopping it up.
   ! Note: Only supports freeall operation, not individual frees.
   type smalloc_type
      real(wp), dimension(:), allocatable :: rmem ! real memory
      integer(long) :: rmem_size ! needed as size(rmem,kind=long) is f2003
      integer(long) :: rhead = 0 ! last location containing useful information
         ! in rmem
      integer, dimension(:), allocatable :: imem ! integer memory
      integer(long) :: imem_size ! needed as size(imem,kind=long) is f2003
      integer(long) :: ihead = 0 ! last location containing useful information
         ! in imem
      type(smalloc_type), pointer :: next_alloc => null()
      type(smalloc_type), pointer :: top_real => null() ! Last page where real
         ! allocation was successful
      type(smalloc_type), pointer :: top_int => null() ! Last page where integer
         ! allocation was successful
!$    integer(omp_lock_kind) :: lock
   end type smalloc_type

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   ! Stack memory allocation type
   type stack_mem_type
      real(wp), dimension(:), allocatable :: mem ! real memory
      integer(long) :: mem_size ! needed as size(mem,kind=long) is f2003
      integer(long) :: head = 0 ! last location containing useful information
      type(stack_mem_type), pointer :: below => null() ! next stack frame down
   end type stack_mem_type

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   ! Data type for storing each node of the factors
   type node_type
      integer :: nelim
      integer :: ndelay
      integer(long) :: rdptr ! entry into (rebuilt) rlist_direct
      integer :: ncpdb ! #contrib. to parent's diag. block
      type(C_PTR) :: gpu_lcol
      real(wp), dimension(:), pointer :: lcol ! values in factors
         ! (will also include unneeded data for any columns delayed from this
         ! node)
      integer, dimension(:), pointer :: perm ! permutation of columns at this
         ! node: perm(i) is column index in expected global elimination order
         ! that is actually eliminated at local elimination index i
         ! Assuming no delays or permutation this will be
         ! sptr(node):sptr(node+1)-1
      ! Following components are used to index directly into contiguous arrays
      ! lcol and perm without taking performance hit for passing pointers
      type(smalloc_type), pointer :: rsmptr, ismptr
      integer(long) :: rsmsa, ismsa
   end type node_type

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !
   ! Data type for temporary stack data that is only needed transiently during
   ! factorise phase
   ! Each instance represents a "page" of memory
   !
   type stack_type
      real(wp), dimension(:), pointer :: val => null() ! generated element
      ! Following components allow us to pass contiguous array val without
      ! taking performance hit for passing pointers
      type(stack_mem_type), pointer :: stptr => null()
      integer(long) :: stsa
   end type stack_type

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !
   ! Data type for per-thread stats. This is amalgamated after end of parallel
   ! section to get info parameters of same name.
   !
   type thread_stats
      integer :: flag = SSIDS_SUCCESS
      integer :: st = 0
      integer :: cuda_error = 0
      integer :: cublas_error = 0
      integer :: maxfront = 0 ! Maximum front size
      integer(long) :: num_factor = 0_long ! Number of entries in factors
      integer(long) :: num_flops = 0_long ! Number of floating point operations
      integer :: num_delay = 0 ! Number of delayed variables
      integer :: num_neg = 0 ! Number of negative pivots
      integer :: num_two = 0 ! Number of 2x2 pivots
      integer :: num_zero = 0 ! Number of zero pivots
   end type thread_stats

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !
   ! This type is used to pass buf around for each thread such that it can
   ! be reallocated independantly
   !
   type real_ptr_type
      real(wp), pointer :: chkptr => null()
      real(wp), dimension(:), allocatable :: val
   end type real_ptr_type

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   !
   ! Data type for control parameters
   !
   type ssids_options
      !
      ! Printing options
      !
      integer :: print_level = 0 ! Controls diagnostic printing.
         ! Possible values are:
         !  < 0: no printing.
         !  0: error and warning messages only.
         !  1: as 0 plus basic diagnostic printing.
         !  > 1: as 1 plus some more detailed diagnostic messages.
         !  > 9999: debug (absolutely everything - really don't use this)
      integer :: unit_diagnostics = 6 ! unit number for diagnostic printing.
         ! Printing is suppressed if unit_diagnostics  <  0.
      integer :: unit_error = 6 ! unit number for error messages.
         ! Printing is suppressed if unit_error  <  0.
      integer :: unit_warning = 6 ! unit number for warning messages.
         ! Printing is suppressed if unit_warning  <  0.

      !
      ! Options used ssids_analyse() and ssids_analyse_coord()
      !
      integer :: ordering = 1 ! controls choice of ordering
         ! 0 Order must be supplied by user
         ! 1 METIS ordering with default settings is used.
         ! 2 Matching with METIS on compressed matrix.
      integer :: nemin = nemin_default ! Min. number of eliminations at a tree
         ! node for amalgamation not to be considered.

      !
      ! Options used by ssids_factor() [both indef+posdef]
      !
      integer :: scaling = 0 ! controls use of scaling. 
         !  <=0: user supplied (or no) scaling
         !    1: Matching-based scaling by Hungarian Algorithm (MC64-like)
         !    2: Matching-based scaling by Auction Algorithm
         !    3: Scaling generated during analyse phase for matching-based order
         !  >=4: Norm equilibriation algorithm (MC77-like)

      !
      ! Options used by ssids_factor() with posdef=.false.
      !
      logical :: action = .true. ! used in indefinite case only.
         ! If true and the matrix is found to be
         ! singular, computation continues with a warning.
         ! Otherwise, terminates with error SSIDS_ERROR_SINGULAR.
      real(wp) :: small = 1e-20_wp ! Minimum pivot size (absolute value of a
         ! pivot must be of size at least small to be accepted).
      real(wp) :: u = 0.01

      !
      ! Options used by ssids_factor() and ssids_solve()
      !
      logical :: use_gpu_factor = .true. ! Use GPU for factor phase if true
         ! or CPU if false
      logical :: use_gpu_solve = .true. ! Use GPU for solve phase if true
         ! or CPU if false
      integer :: presolve = 0 ! If set to a non-zero level, triggers L-factor
         ! optimization for the sake of subsequent multiple solves.
         ! Future releases may offer different levels of optimization.

      !
      ! Undocumented
      !
      integer :: nstream = 1 ! Number of streams to use
      real(wp) :: multiplier = 1.1 ! size to multiply expected memory size by
         ! when doing initial memory allocation to allow for delays.
      type(auction_options) :: auction ! Auction algorithm parameters
      real :: min_loadbalance = 0.8 ! Minimum load balance required when
         ! finding level set used for multiple streams

      !
      ! New and undocumented - FIXME decide whether to document before release
      !
      integer :: cpu_small_subtree_threshold = 100**3 ! Flops below which we
         ! treat a subtree as small and use the single core kernel
      integer :: cpu_task_block_size = 256 ! block size to use for task
         ! generation on larger nodes
      integer :: min_npart = 4 ! minimum number of parts to split tree into
      integer(long) :: max_flops_part = 10**9_long ! maximum number of flops
         ! per part when splitting tree
      integer :: pivot_method = PIVOT_METHOD_APP_AGGRESIVE
         ! Type of pivoting to use on CPU side:
         ! 0 - A posteori pivoting, roll back entire front on pivot failure
         ! 1 - A posteori pivoting, roll back on block column level for failure
         ! 2 - Traditional threshold partial pivoting (serial, inefficient!)
   end type ssids_options

   !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   integer, parameter, public :: BLOCK_SIZE = 8
   integer, parameter, public :: MNF_BLOCKS = 11
   integer, parameter, public :: HOGG_ASSEMBLE_TX = 128
   integer, parameter, public :: HOGG_ASSEMBLE_TY = 8

   integer, parameter, public :: DEBUG_PRINT_LEVEL = 9999 

end module spral_ssids_datatypes
