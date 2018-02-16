#ifndef GLOW_SUPPORT_COMPILER_H
#define GLOW_SUPPORT_COMPILER_H

#if !defined(__has_builtin)
#define __has_builtin(builtin) 0
#endif

#if defined(__GNUC__)
#define GLOW_GNUC_ATLEAST(major, minor, patch)                                 \
  (__GNUC__ >= (major) || __GNUC_MINOR__ >= (minor) ||                         \
   __GNUC_PATCHLEVEL__ >= (patch))
#else
#define GLOW_GNUC_ATLEAST(major, minor, patch) 0
#endif

#define GLOW_ASSERT(e)                                                         \
  ((void)((e) ? ((void)0) : GLOW_ASSERT_IMPL(#e, __FILE__, __LINE__)))
#define GLOW_ASSERT_IMPL(e, file, line)                                        \
  ((void)printf("%s:%u: failed assertion `%s'\n", file, line, e), abort())

#define GLOW_UNREACHABLE(msg)                                                  \
  ((void)printf("%s:%u: %s\n", __FILE__, __LINE__, msg), abort())

#endif // GLOW_SUPPORT_COMPILER_H
