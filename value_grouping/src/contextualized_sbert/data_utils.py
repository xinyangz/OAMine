import random
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from dataclasses_json import dataclass_json
from flashtext import KeywordProcessor

NamedCluster = Tuple[str, List[str]]
EntityContext = namedtuple("EntityContext", ["left_context", "entity", "right_context"])

@dataclass_json
@dataclass
class EntityAnnotation:
  start: int
  end: int
  text: str = ""
  label: str = ""


@dataclass_json
@dataclass
class AnnotatedASIN:
  asin: str
  text: str
  entities: List[EntityAnnotation] = field(default_factory=list)


def match_context(phrases: List[str],
                  docs: List[List[str]],
                  sampling=-1):
  raw_texts = [" ".join(doc) for doc in docs]

  # string matching
  phrase2context: Dict[str, List[EntityContext]] = defaultdict(list)
  kw_processor = KeywordProcessor()
  kw_processor.add_keywords_from_list(phrases)
  for raw_text in raw_texts:
    keywords_found = kw_processor.extract_keywords(raw_text, span_info=True)
    for kw, start, end in keywords_found:
      left_ctx = raw_text[:start].strip()
      right_ctx = raw_text[end:].strip()
      phrase2context[kw].append(EntityContext(left_ctx, kw, right_ctx))
  phrase2context = dict(phrase2context)

  phrase2context_idx = dict()
  all_contexts = []
  for phrase in phrases:
    contexts = phrase2context.get(phrase, [EntityContext("", phrase, "")])
    if sampling > 0 and len(contexts) > sampling:
      contexts = random.sample(contexts, sampling)
    start = len(all_contexts)
    end = start
    for context in contexts:
      all_contexts.append(context)
      end += 1
    phrase2context_idx[phrase] = (start, end)
  return phrase2context_idx, all_contexts

