# Development Maintenance

## Private API and Fallback Cleanup

MHCflurry should keep internal helpers movable. Tests and sibling modules should
not pin underscore-prefixed functions unless the helper is truly private to the
same module.

1. Promote shared helpers before cross-module use.
   If a helper is imported by another module or tested directly, give it a
   public name in its owner module and document the contract in one sentence.

2. Test owner modules, not compatibility facades.
   Tests for moved helpers should import the helper module directly. Compatibility
   shim tests should cover public behavior only: old imports still expose the
   documented entry point and produce the same result.

3. Make fallback paths observable.
   Fallbacks for missing optional tools or hardware probes may return a safe
   default. Fallbacks caused by import errors, bad local refactors, or invalid
   user configuration should fail loudly instead of being caught by broad
   `except Exception` blocks.

4. Keep compatibility shims thin.
   Old module paths may re-export public entry points and delegate unknown
   attributes, but they should not replace `sys.modules` or assert identity with
   the new implementation module.

5. Track remaining private test hooks.
   The remaining direct private imports in tests should be triaged into:
   public helper, same-module implementation detail tested through behavior, or
   compatibility shim slated for removal after the 2.3 release line stabilizes.
