"""What this route refuses, and why the refusal has to name a field."""

class SpecIncomplete(RuntimeError):
    """Metadata the compiler requires that this kernel did not provide.

    Raised with the missing field named, rather than writing a spec that fails
    deeper in the pipeline where the cause is unrecoverable.
    """
