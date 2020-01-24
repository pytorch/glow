/**
 * Copyright (c) Glow Contributors. See CONTRIBUTORS file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef GLOW_SUPPORT_REGISTER_H
#define GLOW_SUPPORT_REGISTER_H

#include <cassert>
#include <map>

namespace glow {

/// Base factory interface which needs to be implemented
/// for static registration of arbitrary classes.
/// For example, CPUFactory would be responsible for creating CPU backends
/// registred with "CPU" key.
template <class Key, class Base> class BaseFactory {
public:
  virtual ~BaseFactory() = default;
  /// Create an object of Base type.
  virtual Base *create() = 0;
  /// Key used for a registered factory.
  virtual Key getRegistrationKey() const = 0;
  /// Number of devices available for the registered factory.
  virtual unsigned numDevices() const = 0;
};

/// General registry for implementation factories.
/// The registry is templated by the Key class and Base class that a
/// set of factories inherits from.
template <class Key, class Base> class FactoryRegistry {
public:
  using FactoryMap = std::map<Key, BaseFactory<Key, Base> *>;

  /// Register \p factory in a static map.
  static void registerFactory(BaseFactory<Key, Base> &factory) {
    Key registrationKey = factory.getRegistrationKey();
    auto inserted = factories().emplace(registrationKey, &factory);
    assert(inserted.second && "Double registration of base factory");
    (void)inserted;
  }

  /// \returns newly created object from factory keyed by \p key.
  /// \returns nullptr if there is no factory registered with \p key.
  static Base *get(const Key &key) {
    auto it = factories().find(key);

    if (it == factories().end()) {
      return nullptr;
    }

    return it->second->create();
  }

  /// \returns all registered factories.
  static FactoryMap &factories() {
    static FactoryMap *factories = new FactoryMap();
    return *factories;
  }
};

/// Factory registration template, all static registration should be done
/// via RegisterFactory. It allows to register specific implementation factory
/// with the FactoryRegistry by instantiating this templated class with the
/// specific factory class, specific Key class and the general Base class.
///
/// Example registration:
/// static Registry::RegisterFactory<
///          SpecificKeyType, SpecificFactory, BaseFactory> registered_;
template <class Key, class Factory, class Base> class RegisterFactory {
public:
  RegisterFactory() { FactoryRegistry<Key, Base>::registerFactory(factory_); }

private:
  Factory factory_{};
};

} // namespace glow

#endif // GLOW_SUPPORT_REGISTER_H
