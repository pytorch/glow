## Memory Regions

Memory regions are used to control the placement of tensors in the target memory.

### Basic memory regions support in Glow

Glow originally supported only three logical memory regions, which are contiguous  blocks of memory containing tensors of specific kinds: activations, constant weights and mutable weights. This coarse-grained approach is simple enough and works pretty well for many use-cases.

But more advanced scenarios may require a more flexible way to combine tensors into logical memory regions and control their properties. For example it may be required for HW targets with more complex memory hierarchies or when a fine-grained control over tensors memory placement is needed.

### Advanced memory regions support in Glow

Glow allows for very flexible and configurable definition of memory regions, which allows for a fine-grained control over tensors placement in memory. It is a generalization of the old Glow mechanism, which had only three default memory regions described above.

Memory regions can be used to group different tensors together to form a logical region of memory and specify various properties for this region of memory, for example:
   * Which tensors should be placed in a given region of memory? E.g. only FC weights, or only input tensors.
   * Which memory allocation algorithm to use for a given region?
   * Can tensors in a given region reuse memory or not? Reuse may make sense for regions containing activations or intermediate results.
   * One could easily build mechanisms to define region matching rules for tensors using predicates and AND/OR conditions to specify which tensors belong to each memory region.
   * Should each tensor matchig the rule be in its own memory region or should they be combined?
   * Which part of the memory hierachy should be used for a memory region, e.g. DDR or faster memories?
   * Which exact address should be used as a starting address of a given memory region.

Descriptions of memory regions could be potentially defined programmatically or e.g. by means of a config file.

The ability to define memory regions is useful e.g. for performance experiments and deployment experiments. For example, a selective placement of specific kinds of tensors at the best-suited levels of the memory hierarchy can significantely improve performance.

#### Memory regions APIs

The following types are used to define memory regions:
   * `MemoryRegionDescription`
       * Description of a memory region to be used for creating an instance of a memory region. It describes the properties of a memory region and also rules defining which logical buffers belong to this memory region. Each `MemoryRegionDescriptionObject` has a `contains` predicate that can decide if a given tensor should belong to a `MemoryRegion` instance based on the current `MemoryRegionDescription`.
       * A set of standard memory regions descriptions corresponding to the old Glow memory regions are provided: `ConstantWeightMemoryRegionDescription`, `MutableWeightMemoryRegionDescription`, `MutableInputWeightMemoryRegionDescription`, `MutableOutputWeightMemoryRegionDescription` and `ActivationMemoryRegionDescription`.

  * `MemoryRegion`
     * A memory region used for allocating objects like weights, activations, placeholders, etc. Multiple tensors can be allocated in the same memory region if a backend allows for it, depending e.g. on the type and kind of those tensors and some backend specifics. Memory region cannot contain symbols that have totally different semantics. E.g. constant weights likely cannot be combined with per-run activations. Each tensor may belong to exactly one region. Each `MemoryRegion` object contains a symbol table and a set of attributes, so that it can be queried about e.g. if a given tensor belongs to this region. It also makes it easy to get a set of all tensors belonging to the current `MemoryRegion`.

  * `RuntimeSymbolInfo`
     * This type is extended to contain a pointer to the `MemoryRegion` the current symbol belongs to.

MemoryRegionDescritions could be defined programmatically. A backend could easily add a possibility to defined them e.g. via a user-specified configuration file.

The `glow::createMemoryRegionTable` method iterates over all tensors used by a model and uses MemoryRegionDescritions to decide which region any given tensor belongs to. If it cannot find an instance of a given region, it will create such an instance. When this method finishes, each tensor in the model is assigned to a one and only one memory region and this is captured in a `MemoryRegionTableTy` object, which maps memory region IDs to the memory region objects.

Most of the describe APIs are defined in `BackendUtils.h` and implemented in `MemoryRegions.cpp`.

### Backwards compatibility

Memory allocation and memory management mechanisms of the compiler are re-forumulated to be memory region based. If no user-defined memory regions definitions are provided by the user, default memory regions definitions are used, which correspond to the old compiler behaviour and are supposed to be backwards compatible.
