namespace Ametrin.Numerics;

[AttributeUsage(AttributeTargets.Method)]
internal sealed class GenerateVariantsAttribute : Attribute;

[AttributeUsage(AttributeTargets.Class)]
internal sealed class NumericsHelperAttribute<T> : Attribute;